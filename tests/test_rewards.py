import json
from pathlib import Path

import pytest

from src.data.common import write_jsonl
from src.data.reward_fixtures import build_cases, create_database
from src.grpo.rewards import COMPONENT_NAMES, group_advantages, run_self_test, score_candidate
from src.utils.config import load_yaml


@pytest.fixture(scope="module")
def reward_cases_path(tmp_path_factory):
    fixture_dir = tmp_path_factory.mktemp("reward-fixtures")
    database = fixture_dir / "reward_test.sqlite"
    cases_path = fixture_dir / "reward_cases.jsonl"
    create_database(database)
    write_jsonl(cases_path, build_cases(load_yaml("configs/data.yaml"), database))
    return cases_path


@pytest.fixture(scope="module")
def config(reward_cases_path):
    config = load_yaml("configs/grpo.yaml")
    config["self_test"]["cases"] = str(reward_cases_path)
    return config


@pytest.fixture(scope="module")
def report(config):
    return run_self_test(config)


def test_all_frozen_reward_cases_pass(report):
    assert report["status"] == "passed"
    assert report["case_count"] == 13
    assert set(report["component_names"]) == set(COMPONENT_NAMES)


def test_gold_and_equivalent_sql_get_same_highest_reward(report):
    rows = {row["id"]: row for row in report["cases"]}
    maximum = max(row["total"] for row in rows.values())
    assert rows["gold_sql"]["total"] == maximum
    assert rows["equivalent_sql"]["total"] == maximum
    assert rows["gold_sql"]["execution_correct"]
    assert rows["equivalent_sql"]["execution_correct"]


def test_error_classes_and_component_separation(report):
    rows = {row["id"]: row for row in report["cases"]}
    assert rows["wrong_result"]["class"] == "wrong_result"
    assert rows["syntax_error"]["class"] == "execution_error"
    assert rows["dangerous_delete"]["class"] == "unsafe_sql"
    assert rows["dangerous_drop"]["class"] == "unsafe_sql"
    assert rows["empty_output"]["class"] == "empty_output"
    assert rows["markdown_wrapped"]["class"] == "format_error"
    assert rows["dangerous_delete"]["components"]["unsafe_sql"] < 0
    assert rows["empty_output"]["components"]["empty_output"] < 0
    assert rows["markdown_wrapped"]["components"]["execution_correctness"] > 0
    assert rows["markdown_wrapped"]["components"]["format_valid"] == 0
    for row in rows.values():
        assert set(row["components"]) == set(COMPONENT_NAMES)
        assert row["total"] == pytest.approx(sum(row["components"].values()))


def test_string_similarity_is_not_used(config):
    case = next(
        json.loads(line)
        for line in Path(config["self_test"]["cases"]).open(encoding="utf-8")
        if json.loads(line)["id"] == "equivalent_sql"
    )
    context = {
        "db_path": case["db_path"],
        "gold_result": case["gold_result"],
        "order_sensitive": case["order_sensitive"],
        "float_tolerance": case["float_tolerance"],
        "timeout_seconds": 3.0,
        "max_result_rows": 1000,
    }
    reward = score_candidate(case["candidate_sql"], context, config)
    assert case["candidate_sql"] != case["gold_sql"]
    assert reward["execution_correct"]


def test_group_advantages():
    advantages = group_advantages([0.0, 1.0, 2.0])
    assert sum(advantages) == pytest.approx(0.0)
    assert advantages[0] < advantages[1] < advantages[2]
    assert group_advantages([1.2, 1.2, 1.2]) == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        group_advantages([])
