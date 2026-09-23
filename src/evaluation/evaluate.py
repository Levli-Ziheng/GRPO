"""Evaluate saved predictions or invoke the shared Base generation pipeline."""
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path
from src.data.common import read_jsonl, write_jsonl
from src.utils.config import load_yaml, write_json
from src.inference.sql_parser import parse_sql
from src.execution.sqlite_executor import execute_query
from src.execution.result_normalizer import results_equal
from src.evaluation.metrics import summarize

def evaluate_record(prediction, task, config):
    parsed = parse_sql(prediction["raw_output"])
    result = dict(prediction)
    result.update({"db_id": task["db_id"], "difficulty": task["difficulty"], "question": task["question"],
        "gold_sql": task["gold_sql"], "gold_result": task["gold_result"],
        "predicted_sql": parsed["parsed_sql"], "format_valid": parsed["format_valid"],
        "execution_success": False, "execution_correct": False, "predicted_result": None,
        "execution_error": parsed["parse_error"], "error_type": "format_error", "execution_seconds": 0})
    if parsed["parsed_sql"]:
        try:
            executed = execute_query(task["db_path"], parsed["parsed_sql"],
                timeout_seconds=config["timeout_seconds"], max_result_rows=config["max_result_rows"])
            result.update({"execution_success": True, "predicted_result": executed["rows"],
                           "predicted_column_count": executed["column_count"], "execution_seconds": executed["elapsed_seconds"]})
            result["execution_correct"] = results_equal(executed["rows"], task["gold_result"],
                order_sensitive=task["order_sensitive"], tolerance=config["float_tolerance"],
                predicted_columns=executed["column_count"], gold_columns=task["gold_result_column_count"])
            result["error_type"] = None if result["execution_correct"] else "result_mismatch"
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
            result.update({"execution_error": str(exc), "error_type": "execution_error"})
    return result

def write_evaluation(out, predictions):
    metrics = summarize(predictions)
    write_json(out / "evaluation_metrics.json", metrics)
    write_jsonl(out / "error_cases.jsonl", (r for r in predictions if not r["execution_correct"] or not r["format_valid"]))
    write_jsonl(out / "success_cases.jsonl", [r for r in predictions if r["execution_correct"]][:3])
    return metrics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--stage", choices=["base"], default="base")
    parser.add_argument("--output-dir")
    parser.add_argument("--predictions")
    args = parser.parse_args()
    if not args.predictions:
        from src.inference.generate import run
        args.mode, args.question, args.schema_file = "batch", None, None
        return run(args)
    from src.inference.generate import load_tasks
    cfg = load_yaml(args.config)
    tasks = load_tasks(cfg, load_yaml(cfg["data_config"]))
    predictions = list(read_jsonl(args.predictions))
    if [r["id"] for r in predictions] != [t["id"] for t in tasks]:
        raise ValueError("Prediction IDs must exactly match the entire locked test set in order")
    if not args.output_dir:
        parser.error("--output-dir is required for saved prediction evaluation")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    evaluated = [evaluate_record(r, t, cfg["evaluation"]) for r, t in zip(predictions, tasks)]
    write_jsonl(out / "predictions.jsonl", evaluated)
    print(json.dumps(write_evaluation(out, evaluated), ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
