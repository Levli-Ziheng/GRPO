from __future__ import annotations

import sqlite3

import pytest

from src.data.common import execute_readonly, readonly_sql_reason


@pytest.fixture()
def database(tmp_path):
    path = tmp_path / "test.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE item(id INTEGER PRIMARY KEY, value REAL);"
        "INSERT INTO item VALUES (1, 1.0), (2, 2.0);"
    )
    connection.commit()
    connection.close()
    return path


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT * FROM item", None),
        ("WITH x AS (SELECT * FROM item) SELECT * FROM x", None),
        ("", "empty_sql"),
        ("DELETE FROM item", "not_select_or_with"),
        ("SELECT 1; SELECT 2", "multiple_statements"),
        ("WITH x AS (DELETE FROM item RETURNING *) SELECT * FROM x", "forbidden_keyword:DELETE"),
    ],
)
def test_readonly_sql_reason(sql, expected):
    assert readonly_sql_reason(sql) == expected


def test_execute_readonly_normalizes_unordered_results(database):
    report = execute_readonly(
        database,
        "SELECT value FROM item ORDER BY id DESC",
        timeout_seconds=1.0,
        max_result_rows=10,
    )
    assert report["rows"] == [[2.0], [1.0]]
    assert report["order_sensitive"] is True


def test_execute_readonly_rejects_writes(database):
    with pytest.raises(ValueError):
        execute_readonly(
            database,
            "DROP TABLE item",
            timeout_seconds=1.0,
            max_result_rows=10,
        )
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT COUNT(*) FROM item").fetchone()[0] == 2
    connection.close()
