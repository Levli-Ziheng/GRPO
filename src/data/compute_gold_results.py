from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.common import execute_readonly, read_jsonl, stable_hash, write_jsonl
from src.utils.config import load_yaml, write_json


def failure_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "interrupted" in message:
        return "timeout"
    if "no such table" in message:
        return "missing_table"
    if "no such column" in message:
        return "missing_column"
    if "syntax error" in message or "incomplete input" in message:
        return "syntax_error"
    if "row_limit" in message:
        return "result_too_large"
    if isinstance(exc, ValueError):
        return "unsafe_or_invalid_sql"
    if isinstance(exc, sqlite3.Error):
        return "sqlite_error"
    return "other"


def execute_all(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    execution = config["execution"]
    successful: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    codes: Counter[str] = Counter()
    records = list(read_jsonl(config["paths"]["canonical_all"]))

    for index, record in enumerate(records, start=1):
        try:
            result = execute_readonly(
                record["db_path"],
                record["gold_sql"],
                timeout_seconds=float(execution["timeout_seconds"]),
                max_result_rows=int(execution["max_result_rows"]),
            )
            enriched = dict(record)
            enriched.update(
                {
                    "gold_result": result["rows"],
                    "gold_result_hash": stable_hash(result["rows"]),
                    "result_compare_mode": (
                        "ordered" if result["order_sensitive"] else "unordered"
                    ),
                    "order_sensitive": result["order_sensitive"],
                    "float_tolerance": float(execution["float_tolerance"]),
                    "gold_result_row_count": result["row_count"],
                    "gold_result_column_count": result["column_count"],
                    "gold_execution_ms": result["elapsed_ms"],
                    "gold_execution_success": True,
                    "empty_result": result["row_count"] == 0,
                }
            )
            successful.append(enriched)
        except Exception as exc:  # every rejected public row must be audited
            code = failure_code(exc)
            codes[code] += 1
            failures.append(
                {
                    "id": record["id"],
                    "db_id": record["db_id"],
                    "gold_sql": record["gold_sql"],
                    "reason_code": code,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if index % 500 == 0 or index == len(records):
            print(
                f"Gold execution: {index}/{len(records)}; "
                f"success={len(successful)}, failure={len(failures)}"
            )

    summary = {
        "input_records": len(records),
        "successful_records": len(successful),
        "failed_records": len(failures),
        "success_rate": round(len(successful) / max(len(records), 1), 6),
        "failure_codes": dict(sorted(codes.items())),
        "empty_result_records": sum(row["empty_result"] for row in successful),
    }
    return successful, failures, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute and cache every CSpider Gold SQL.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    successful, failures, summary = execute_all(config)
    write_jsonl(config["paths"]["canonical_executed"], successful)
    write_jsonl(config["paths"]["execution_failures"], failures)
    write_json(config["paths"]["execution_summary"], summary)
    print(
        f"Gold SQL complete: {summary['successful_records']} success, "
        f"{summary['failed_records']} rejected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
