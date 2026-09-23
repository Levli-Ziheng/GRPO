"""Read-only bounded execution. Preserve returned order for gold-driven comparison."""
from __future__ import annotations
import sqlite3
import time
from pathlib import Path
from src.data.common import normalize_cell
from src.inference.sql_parser import query_reason

def execute_query(db_path, sql, *, timeout_seconds=3.0, max_result_rows=1000):
    reason = query_reason(sql)
    if reason:
        raise ValueError(reason)
    path = Path(db_path).resolve(strict=True)
    started = time.perf_counter()
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=timeout_seconds)
    try:
        connection.execute("PRAGMA query_only=ON")
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        connection.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
        connection.set_progress_handler(lambda: int(time.perf_counter() - started > timeout_seconds), 1000)
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(max_result_rows + 1)
        if len(rows) > max_result_rows:
            raise RuntimeError("result_row_limit_exceeded")
        return {"rows": [[normalize_cell(c) for c in row] for row in rows],
                "column_count": len(cursor.description or ()),
                "elapsed_seconds": time.perf_counter() - started}
    finally:
        connection.close()
