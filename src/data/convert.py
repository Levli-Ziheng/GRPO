from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from src.data.common import load_json, normalized_sql, write_jsonl
from src.utils.config import load_yaml, write_json


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def serialize_schema(table: dict[str, Any]) -> str:
    table_names = table["table_names_original"]
    columns = table["column_names_original"]
    column_types = table["column_types"]
    primary_keys = set(table.get("primary_keys", []))
    foreign_keys = table.get("foreign_keys", [])
    per_table: dict[int, list[int]] = {index: [] for index in range(len(table_names))}
    for column_index, (table_index, _column_name) in enumerate(columns):
        if table_index >= 0:
            per_table[table_index].append(column_index)

    statements: list[str] = []
    for table_index, table_name in enumerate(table_names):
        definitions: list[str] = []
        for column_index in per_table[table_index]:
            _, column_name = columns[column_index]
            column_type = str(column_types[column_index]).upper()
            if column_type == "NUMBER":
                column_type = "NUMERIC"
            definition = f"  {quote_identifier(column_name)} {column_type}"
            if column_index in primary_keys:
                definition += " PRIMARY KEY"
            definitions.append(definition)
        for source_index, target_index in foreign_keys:
            source_table, source_column = columns[source_index]
            target_table, target_column = columns[target_index]
            if source_table != table_index:
                continue
            definitions.append(
                f"  FOREIGN KEY ({quote_identifier(source_column)}) REFERENCES "
                f"{quote_identifier(table_names[target_table])} "
                f"({quote_identifier(target_column)})"
            )
        body = ",\n".join(definitions)
        statements.append(f"CREATE TABLE {quote_identifier(table_name)} (\n{body}\n);")
    return "\n\n".join(statements)


def sql_features(sql: str) -> list[str]:
    upper = sql.upper()
    checks = [
        ("join", r"\bJOIN\b"),
        ("filter", r"\bWHERE\b"),
        ("aggregate", r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\("),
        ("group", r"\bGROUP\s+BY\b"),
        ("having", r"\bHAVING\b"),
        ("order", r"\bORDER\s+BY\b"),
        ("limit", r"\bLIMIT\b"),
        ("set_operation", r"\b(UNION|INTERSECT|EXCEPT)\b"),
        ("subquery", r"\(\s*SELECT\b"),
        ("distinct", r"\bDISTINCT\b"),
    ]
    return [name for name, pattern in checks if re.search(pattern, upper)]


def difficulty(sql: str, features: list[str]) -> str:
    joins = len(re.findall(r"\bJOIN\b", sql, flags=re.IGNORECASE))
    score = len(features) + joins
    if "subquery" in features or "set_operation" in features:
        score += 2
    if score <= 2:
        return "easy"
    if score <= 4:
        return "medium"
    if score <= 6:
        return "hard"
    return "extra"


def convert_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = config["source"]
    paths = config["paths"]
    tables = load_json(source["tables_json"])
    table_map = {item["db_id"]: item for item in tables}
    source_manifest = load_json(paths["source_manifest"])
    database_manifest = source_manifest["databases"]
    output: list[dict[str, Any]] = []
    schema_cache: dict[str, str] = {}

    for source_split, source_path in (
        ("train", source["train_json"]),
        ("dev", source["dev_json"]),
    ):
        rows = load_json(source_path)
        for index, row in enumerate(rows):
            db_id = row["db_id"]
            if db_id not in table_map or db_id not in database_manifest:
                raise ValueError(f"Missing schema/database for db_id={db_id}")
            schema = schema_cache.setdefault(db_id, serialize_schema(table_map[db_id]))
            query = normalized_sql(row["query"])
            features = sql_features(query)
            output.append(
                {
                    "id": f"{source['name']}_{source_split}_{index:06d}",
                    "source": source["name"],
                    "source_version": source["version"],
                    "source_split": source_split,
                    "source_index": index,
                    "question": row["question"].strip(),
                    "db_id": db_id,
                    "db_path": database_manifest[db_id]["path"],
                    "db_sha256": database_manifest[db_id]["sha256"],
                    "schema": schema,
                    "gold_sql": query,
                    "difficulty": difficulty(query, features),
                    "difficulty_source": "heuristic_v1",
                    "sql_features": features,
                }
            )

    summary = {
        "records": len(output),
        "train_records": sum(row["source_split"] == "train" for row in output),
        "dev_records": sum(row["source_split"] == "dev" for row in output),
        "databases": len({row["db_id"] for row in output}),
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert CSpider to canonical JSONL.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    config = load_yaml(args.config)
    rows, summary = convert_records(config)
    output = Path(config["paths"]["canonical_all"])
    count = write_jsonl(output, rows)
    write_json(output.with_suffix(".summary.json"), summary)
    print(f"Converted {count} canonical records -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
