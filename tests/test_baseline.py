import sqlite3
import pytest
from src.inference.sql_parser import parse_sql, query_reason
from src.execution.sqlite_executor import execute_query
from src.execution.result_normalizer import results_equal
from src.evaluation.metrics import summarize

def test_parser():
    assert parse_sql(" SELECT 1; ")["format_valid"]
    assert not parse_sql("```sql\nSELECT 1;\n```")["format_valid"]
    assert parse_sql("```sql\nSELECT 1;\n```")["parsed_sql"] == "SELECT 1;"
    for text in ("", "Here is SQL: SELECT 1", "SELECT 1; DELETE FROM t", "WITH x AS (DELETE FROM t) SELECT * FROM x"):
        assert not parse_sql(text)["parsed_sql"]
    assert query_reason("SELECT 'DROP; TABLE', \"update\" FROM t") is None
    assert query_reason("SELECT 1; -- hi") is None

def test_compare():
    assert results_equal([[2],[1]], [[1],[2]], order_sensitive=False)
    assert not results_equal([[2],[1]], [[1],[2]], order_sensitive=True)
    assert not results_equal([[1],[1]], [[1],[2]], order_sensitive=False)
    assert not results_equal([[None]], [[0]], order_sensitive=False)
    assert not results_equal([["1"]], [[1]], order_sensitive=False)
    assert results_equal([[1.0000001]], [[1]], order_sensitive=False)
    assert not results_equal([], [], order_sensitive=False, predicted_columns=1, gold_columns=2)
    assert results_equal([[0.6e-6],[0]], [[0],[1.5e-6]], order_sensitive=False)
    assert not results_equal([[1,2]], [[2,1]], order_sensitive=False)

def test_executor(tmp_path):
    path = tmp_path / "test.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript("CREATE TABLE t(x); INSERT INTO t VALUES(1),(2);")
    assert execute_query(path, "SELECT x FROM t ORDER BY x DESC")["rows"] == [[2],[1]]
    assert execute_query(path, "SELECT 'DELETE;';")["rows"] == [["DELETE;"]]
    assert execute_query(path, "SELECT * FROM t WHERE 0")["column_count"] == 1
    for sql in ("DROP TABLE t", "SELECT 1; SELECT 2", "SELECT load_extension('bad')"):
        with pytest.raises((ValueError, sqlite3.Error)):
            execute_query(path, sql)
    with pytest.raises(RuntimeError):
        execute_query(path, "SELECT * FROM t", max_result_rows=1)
    with pytest.raises(sqlite3.OperationalError):
        execute_query(path, "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n", timeout_seconds=0.01)
    assert execute_query(path, "SELECT COUNT(*) FROM t")["rows"] == [[2]]

def test_metrics_denominator():
    base = dict(format_valid=True, execution_success=True, execution_correct=True,
                difficulty="easy", input_tokens=10, output_tokens=2,
                latency_seconds=1.0, truncated=False, error_type=None)
    bad = dict(base, execution_success=False, execution_correct=False,
               difficulty="hard", error_type="execution_error")
    report = summarize([base, bad])
    assert report["execution_accuracy"] == 0.5
    assert report["by_difficulty"]["hard"]["execution_accuracy"] == 0
    assert report["error_counts"] == {"execution_error": 1}

def test_evaluate_equivalent_sql(tmp_path):
    from src.evaluation.evaluate import evaluate_record
    path = tmp_path / "equiv.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript("CREATE TABLE t(x); INSERT INTO t VALUES(1),(2);")
    task = dict(db_id="test", difficulty="easy", question="How many?",
                db_path=str(path), gold_sql="SELECT count(*) FROM t",
                gold_result=[[2]], order_sensitive=False, gold_result_column_count=1)
    config = dict(timeout_seconds=1, max_result_rows=10, float_tolerance=1e-6)
    assert evaluate_record(dict(raw_output="SELECT SUM(1) FROM t"), task, config)["execution_correct"]
    assert not evaluate_record(dict(raw_output="SELECT 1"), task, config)["execution_correct"]
    assert not evaluate_record(dict(raw_output="SELECT missing FROM t"), task, config)["execution_success"]
