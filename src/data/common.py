from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|VACUUM|PRAGMA|REINDEX|ANALYZE)\b",
    re.IGNORECASE,
)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(output)
    return count


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip()


def readonly_sql_reason(sql: str) -> str | None:
    candidate = sql.strip()
    if not candidate:
        return "empty_sql"
    if "\x00" in candidate:
        return "nul_byte"
    without_trailing = candidate.rstrip().rstrip(";").rstrip()
    if ";" in without_trailing:
        return "multiple_statements"
    first = re.match(r"^(SELECT|WITH)\b", without_trailing, flags=re.IGNORECASE)
    if not first:
        return "not_select_or_with"
    forbidden = FORBIDDEN_SQL.search(without_trailing)
    if forbidden:
        return f"forbidden_keyword:{forbidden.group(1).upper()}"
    return None


def normalize_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value == 0:
            return 0.0
        return float(f"{value:.12g}")
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    return str(value)


def normalize_result(rows: list[tuple[Any, ...]], order_sensitive: bool) -> list[list[Any]]:
    normalized = [[normalize_cell(value) for value in row] for row in rows]
    if not order_sensitive:
        normalized.sort(
            key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True)
        )
    return normalized


def sql_order_sensitive(sql: str) -> bool:
    return bool(re.search(r"\bORDER\s+BY\b", sql, flags=re.IGNORECASE))


def execute_readonly(
    db_path: str | Path,
    sql: str,
    *,
    timeout_seconds: float,
    max_result_rows: int,
) -> dict[str, Any]:
    reason = readonly_sql_reason(sql)
    if reason:
        raise ValueError(reason)

    database = Path(db_path).resolve()
    uri = f"file:{database.as_posix()}?mode=ro"
    started = time.perf_counter()
    deadline = started + timeout_seconds
    connection = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    try:
        connection.execute("PRAGMA query_only = ON")

        def interrupt_if_expired() -> int:
            return 1 if time.perf_counter() > deadline else 0

        connection.set_progress_handler(interrupt_if_expired, 10_000)
        cursor = connection.execute(sql)
        raw_rows = cursor.fetchmany(max_result_rows + 1)
        if len(raw_rows) > max_result_rows:
            raise RuntimeError("result_row_limit_exceeded")
        order_sensitive = sql_order_sensitive(sql)
        result = normalize_result(raw_rows, order_sensitive)
        return {
            "rows": result,
            "row_count": len(result),
            "column_count": len(cursor.description or []),
            "order_sensitive": order_sensitive,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    finally:
        connection.close()


def build_messages(config: dict[str, Any], schema: str, question: str) -> list[dict[str, str]]:
    prompt = config["prompt"]
    user = prompt["user_template"].format(schema=schema, question=question)
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": user},
    ]


def percentile(values: list[int | float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(float(ordered[lower] * (1 - weight) + ordered[upper] * weight), 2)
