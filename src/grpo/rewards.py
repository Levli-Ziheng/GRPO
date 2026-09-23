"""Execution-based Text-to-SQL rewards with explicit component reporting."""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from src.data.common import read_jsonl
from src.execution.result_normalizer import cell_equal, results_equal
from src.execution.sqlite_executor import execute_query
from src.inference.sql_parser import parse_sql, sql_code
from src.utils.config import load_yaml, write_json


COMPONENT_NAMES = (
    "execution_correctness",
    "partial_result",
    "format_valid",
    "execution_success",
    "unsafe_sql",
    "empty_output",
)
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"VACUUM|PRAGMA|REINDEX|ANALYZE)\b",
    re.IGNORECASE,
)


def _candidate_text(raw_output: str) -> str:
    text = raw_output.strip()
    fenced = re.fullmatch(r"```(?:sql|sqlite)?\s*\n?([\s\S]*?)\n?\s*```", text, re.I)
    return fenced.group(1).strip() if fenced else text


def _unsafe_reason(candidate: str, parse_error: str | None) -> str | None:
    code = sql_code(candidate).strip()
    forbidden = _FORBIDDEN.search(code)
    if forbidden:
        return f"forbidden_keyword:{forbidden.group().upper()}"
    if parse_error == "multiple_statements":
        return parse_error
    return None


def _weights(config: dict[str, Any]) -> dict[str, float]:
    weights = config["reward"]["weights"]
    if set(weights) != set(COMPONENT_NAMES):
        raise ValueError(f"Reward weights must be exactly {COMPONENT_NAMES}")
    values = {name: float(weights[name]) for name in COMPONENT_NAMES}
    if values["execution_correctness"] <= 0:
        raise ValueError("Execution correctness must have positive weight")
    if not 0 < values["partial_result"] < values["execution_correctness"]:
        raise ValueError("Partial-result weight must be positive and below correctness")
    if values["unsafe_sql"] >= 0 or values["empty_output"] >= 0:
        raise ValueError("Unsafe and empty output weights must be penalties")
    return values


def _ratio_similarity(left: int, right: int) -> float:
    return 1.0 - abs(left - right) / max(left, right, 1)


def partial_result_similarity(
    predicted: list[list[Any]],
    gold: list[list[Any]],
    *,
    order_sensitive: bool,
    tolerance: float,
    predicted_columns: int,
    gold_columns: int,
) -> float:
    """Grade incorrect executable results without comparing SQL text.

    Exact row overlap remains dominant. Row-count and column-count similarity
    provide a bounded signal when no complete row matches. The score is in
    [0, 1], and callers apply it only after exact execution correctness fails.
    """
    def row_equal(left, right):
        return len(left) == len(right) and all(
            cell_equal(a, b, tolerance) for a, b in zip(left, right)
        )

    if order_sensitive:
        matches = sum(row_equal(a, b) for a, b in zip(predicted, gold))
    else:
        edges = [[j for j, b in enumerate(gold) if row_equal(a, b)] for a in predicted]
        matched_right: dict[int, int] = {}
        for left in range(len(predicted)):
            visited = set()
            def augment(index):
                for right in edges[index]:
                    if right in visited:
                        continue
                    visited.add(right)
                    if right not in matched_right or augment(matched_right[right]):
                        matched_right[right] = index
                        return True
                return False
            augment(left)
        matches = len(matched_right)

    row_f1 = 0.0 if not predicted or not gold else 2.0 * matches / (len(predicted) + len(gold))
    row_count = _ratio_similarity(len(predicted), len(gold))
    column_count = _ratio_similarity(predicted_columns, gold_columns)
    score = 0.6 * row_f1 + 0.2 * row_count + 0.2 * column_count
    return min(1.0, max(0.0, float(score)))


def score_candidate(
    raw_output: str,
    reward_context: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Score one model output without using SQL string similarity."""
    weights = _weights(config)
    components = {name: 0.0 for name in COMPONENT_NAMES}
    parsed = parse_sql(raw_output)
    candidate = _candidate_text(raw_output)
    unsafe_reason = _unsafe_reason(candidate, parsed["parse_error"])
    result: dict[str, Any] = {
        "class": None,
        "raw_output": raw_output,
        "parsed_sql": parsed["parsed_sql"],
        "format_valid": bool(parsed["format_valid"]),
        "execution_success": False,
        "execution_correct": False,
        "execution_error": parsed["parse_error"],
        "unsafe_reason": unsafe_reason,
        "predicted_result": None,
        "components": components,
    }

    if not candidate:
        result["class"] = "empty_output"
        components["empty_output"] = weights["empty_output"]
    elif unsafe_reason:
        result["class"] = "unsafe_sql"
        components["unsafe_sql"] = weights["unsafe_sql"]
    elif not parsed["parsed_sql"]:
        result["class"] = "execution_error"
    else:
        if parsed["format_valid"]:
            components["format_valid"] = weights["format_valid"]
        try:
            executed = execute_query(
                reward_context["db_path"],
                parsed["parsed_sql"],
                timeout_seconds=float(reward_context["timeout_seconds"]),
                max_result_rows=int(reward_context["max_result_rows"]),
            )
            result.update(
                execution_success=True,
                execution_error=None,
                predicted_result=executed["rows"],
                predicted_column_count=executed["column_count"],
                execution_seconds=executed["elapsed_seconds"],
            )
            components["execution_success"] = weights["execution_success"]
            gold = reward_context["gold_result"]
            gold_columns = reward_context.get("gold_result_column_count")
            if gold_columns is None and gold:
                gold_columns = len(gold[0])
            correct = results_equal(
                executed["rows"],
                gold,
                order_sensitive=bool(reward_context["order_sensitive"]),
                tolerance=float(reward_context["float_tolerance"]),
                predicted_columns=executed["column_count"],
                gold_columns=gold_columns,
            )
            result["execution_correct"] = bool(correct)
            if correct:
                components["execution_correctness"] = weights["execution_correctness"]
            else:
                similarity = partial_result_similarity(
                    executed["rows"], gold,
                    order_sensitive=bool(reward_context["order_sensitive"]),
                    tolerance=float(reward_context["float_tolerance"]),
                    predicted_columns=executed["column_count"],
                    gold_columns=int(gold_columns or 0),
                )
                result["partial_result_similarity"] = similarity
                components["partial_result"] = weights["partial_result"] * similarity
            if not parsed["format_valid"]:
                result["class"] = "format_error"
            elif correct and not gold:
                result["class"] = "correct_but_empty"
            elif correct:
                result["class"] = "correct"
            else:
                result["class"] = "wrong_result"
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
            result["class"] = "execution_error"
            result["execution_error"] = str(exc)

    result["total"] = float(sum(components.values()))
    return result


def group_advantages(rewards: list[float], epsilon: float = 1e-8) -> list[float]:
    if not rewards:
        raise ValueError("At least one reward is required")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    std = math.sqrt(variance)
    return [(value - mean) / (std + epsilon) for value in rewards]


def run_self_test(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["self_test"]["cases"])
    cases = list(read_jsonl(path))
    expected_count = int(config["self_test"]["expected_cases"])
    if len(cases) != expected_count:
        raise AssertionError(f"Expected {expected_count} cases, found {len(cases)}")

    scored = []
    for case in cases:
        context = {
            "db_path": case["db_path"],
            "gold_result": case["gold_result"],
            "order_sensitive": case["order_sensitive"],
            "float_tolerance": case["float_tolerance"],
            "timeout_seconds": 3.0,
            "max_result_rows": 1000,
        }
        reward = score_candidate(case["candidate_sql"], context, config)
        if reward["class"] != case["expected_class"]:
            raise AssertionError(
                f"{case['id']}: expected {case['expected_class']}, got {reward['class']}"
            )
        scored.append({"id": case["id"], **reward})

    by_id = {row["id"]: row for row in scored}
    maximum = max(row["total"] for row in scored)
    for case_id in ("gold_sql", "equivalent_sql", "ordered_correct", "float_tolerance", "empty_gold_result"):
        if by_id[case_id]["total"] != maximum or not by_id[case_id]["execution_correct"]:
            raise AssertionError(f"{case_id} did not receive the highest correct reward")
    if any(abs(value) > 1e-12 for value in group_advantages([maximum] * 4)):
        raise AssertionError("Equal-reward group should have zero advantages")

    return {
        "status": "passed",
        "case_count": len(scored),
        "class_counts": {
            name: sum(row["class"] == name for row in scored)
            for name in sorted({row["class"] for row in scored})
        },
        "highest_reward": maximum,
        "component_names": list(COMPONENT_NAMES),
        "cases": scored,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Only --self-test is available in stage 6")
    report = run_self_test(load_yaml(args.config))
    if args.output:
        write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("status", "case_count", "class_counts", "highest_reward", "component_names")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
