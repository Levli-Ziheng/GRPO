from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

from src.data.common import execute_readonly, stable_hash, write_jsonl
from src.utils.config import load_yaml, write_json


def create_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".sqlite.tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            CREATE TABLE department (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                budget REAL,
                city TEXT
            );
            CREATE TABLE employee (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department_id INTEGER,
                salary REAL,
                FOREIGN KEY (department_id) REFERENCES department(id)
            );
            INSERT INTO department VALUES
                (1, '研发', 1500000.0, '杭州'),
                (2, '销售', 900000.0, '上海'),
                (3, '运营', NULL, '杭州');
            INSERT INTO employee VALUES
                (1, '甲', 1, 30000.0),
                (2, '乙', 1, 28000.0),
                (3, '丙', 2, 25000.0),
                (4, '丁', 2, 25000.0000001);
            """
        )
        connection.commit()
    finally:
        connection.close()
    temporary.replace(path)


def build_cases(config: dict[str, Any], database: Path) -> list[dict[str, Any]]:
    execution = config["execution"]
    gold_queries = {
        "count_high_budget": "SELECT COUNT(*) FROM department WHERE budget > 1000000",
        "ordered_departments": "SELECT name FROM department ORDER BY id",
        "salary_values": "SELECT salary FROM employee WHERE department_id = 2 ORDER BY id",
        "empty_city": "SELECT name FROM department WHERE city = '北京'",
    }
    contexts: dict[str, dict[str, Any]] = {}
    for name, sql in gold_queries.items():
        result = execute_readonly(
            database,
            sql,
            timeout_seconds=float(execution["timeout_seconds"]),
            max_result_rows=int(execution["max_result_rows"]),
        )
        contexts[name] = {
            "gold_sql": sql,
            "gold_result": result["rows"],
            "gold_result_hash": stable_hash(result["rows"]),
            "order_sensitive": result["order_sensitive"],
        }

    definitions = [
        ("gold_sql", "count_high_budget", gold_queries["count_high_budget"], "correct"),
        ("equivalent_sql", "count_high_budget", "SELECT COUNT(id) FROM department WHERE 1000000 < budget", "correct"),
        ("wrong_result", "count_high_budget", "SELECT COUNT(*) FROM department", "wrong_result"),
        ("syntax_error", "count_high_budget", "SELEC COUNT(*) FROM department", "execution_error"),
        ("dangerous_delete", "count_high_budget", "DELETE FROM department", "unsafe_sql"),
        ("dangerous_drop", "count_high_budget", "DROP TABLE department", "unsafe_sql"),
        ("empty_output", "count_high_budget", "", "empty_output"),
        ("markdown_wrapped", "count_high_budget", "```sql\nSELECT COUNT(*) FROM department WHERE budget > 1000000\n```", "format_error"),
        ("multiple_statements", "count_high_budget", "SELECT COUNT(*) FROM department; SELECT 1", "unsafe_sql"),
        ("ordered_correct", "ordered_departments", gold_queries["ordered_departments"], "correct"),
        ("ordered_wrong", "ordered_departments", "SELECT name FROM department ORDER BY id DESC", "wrong_result"),
        ("float_tolerance", "salary_values", gold_queries["salary_values"], "correct"),
        ("empty_gold_result", "empty_city", gold_queries["empty_city"], "correct_but_empty"),
    ]
    rows = []
    for case_id, context_name, candidate_sql, expected in definitions:
        rows.append(
            {
                "id": case_id,
                "db_path": database.as_posix(),
                "candidate_sql": candidate_sql,
                "expected_class": expected,
                "float_tolerance": float(execution["float_tolerance"]),
                **contexts[context_name],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic GRPO reward fixtures.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    fixture_dir = Path(config["paths"]["fixtures_dir"])
    database = fixture_dir / "reward_test.sqlite"
    cases_path = fixture_dir / "reward_cases.jsonl"
    create_database(database)
    cases = build_cases(config, database)
    write_jsonl(cases_path, cases)
    report = {
        "database": database.as_posix(),
        "cases": len(cases),
        "expected_classes": sorted({row["expected_class"] for row in cases}),
    }
    write_json(fixture_dir / "reward_fixture_manifest.json", report)
    print(f"Created reward fixtures: {len(cases)} cases -> {cases_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
