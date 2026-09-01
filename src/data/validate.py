from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.common import load_json, percentile, read_jsonl, sha256_file
from src.utils.config import load_yaml, write_json


def load_rows(directory: Path, name: str) -> list[dict[str, Any]]:
    return list(read_jsonl(directory / name))


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row[key]) for row in rows)
    return dict(sorted(counts.items()))


def validate_dataset(config: dict[str, Any], mode: str) -> tuple[dict[str, Any], list[str]]:
    size_config = config["split"][mode]
    version = size_config["version"]
    directory = Path(config["paths"]["processed_root"]) / version
    manifest = load_json(directory / "dataset_manifest.json")
    canonical_train = load_rows(directory, "canonical_train.jsonl")
    canonical_validation = load_rows(directory, "canonical_validation.jsonl")
    canonical_test = load_rows(directory, "canonical_test.jsonl")
    sft_train = load_rows(directory, "sft_train.jsonl")
    sft_validation = load_rows(directory, "sft_validation.jsonl")
    grpo_train = load_rows(directory, "grpo_train.jsonl")
    eval_validation = load_rows(directory, "eval_validation.jsonl")
    eval_test = load_rows(directory, "eval_test.jsonl")
    errors: list[str] = []

    expected_counts = {
        "canonical_train.jsonl": int(size_config["sft_train"]),
        "canonical_validation.jsonl": int(size_config["validation"]),
        "canonical_test.jsonl": int(size_config["test"]),
        "sft_train.jsonl": int(size_config["sft_train"]),
        "sft_validation.jsonl": int(size_config["validation"]),
        "grpo_train.jsonl": int(size_config["grpo_train"]),
        "eval_validation.jsonl": int(size_config["validation"]),
        "eval_test.jsonl": int(size_config["test"]),
    }
    for name, expected in expected_counts.items():
        actual = manifest["counts"].get(name)
        if actual != expected:
            errors.append(f"count:{name}: expected={expected}, actual={actual}")

    split_rows = {
        "train": canonical_train,
        "validation": canonical_validation,
        "test": canonical_test,
    }
    db_sets = {name: {row["db_id"] for row in rows} for name, rows in split_rows.items()}
    id_sets = {name: {row["id"] for row in rows} for name, rows in split_rows.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        db_overlap = sorted(db_sets[left] & db_sets[right])
        id_overlap = sorted(id_sets[left] & id_sets[right])
        if db_overlap:
            errors.append(f"db_leakage:{left}/{right}:{db_overlap}")
        if id_overlap:
            errors.append(f"id_leakage:{left}/{right}:{id_overlap[:5]}")

    max_length = int(config["tokenizer"]["max_sequence_length"])
    for split_name, rows in split_rows.items():
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate_ids:{split_name}")
        for row in rows:
            if not row.get("gold_execution_success"):
                errors.append(f"unexecuted_gold:{row['id']}")
            if int(row["sft_sequence_tokens"]) > max_length:
                errors.append(f"sft_too_long:{row['id']}")
            if int(row["grpo_max_sequence_tokens"]) > max_length:
                errors.append(f"grpo_too_long:{row['id']}")
            if not Path(row["db_path"]).is_file():
                errors.append(f"missing_database:{row['id']}:{row['db_path']}")

    for row in sft_train + sft_validation:
        roles = [message.get("role") for message in row.get("messages", [])]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"invalid_sft_messages:{row['id']}:{roles}")
        if not row["messages"][-1].get("content", "").strip():
            errors.append(f"empty_sft_label:{row['id']}")

    for row in grpo_train:
        if "gold_sql" in row or "gold_result" in row:
            errors.append(f"grpo_top_level_gold_leak:{row['id']}")
        if "gold_sql" in row.get("reward_context", {}):
            errors.append(f"grpo_reward_gold_sql_leak:{row['id']}")
        if not row.get("prompt") or row["prompt"][-1].get("role") != "user":
            errors.append(f"invalid_grpo_prompt:{row['id']}")
        if not row["reward_context"].get("gold_result_hash"):
            errors.append(f"missing_grpo_reward_context:{row['id']}")
        if row["reward_context"].get("gold_result") == []:
            errors.append(f"empty_grpo_gold_result:{row['id']}")

    for name, expected_hash in manifest["file_sha256"].items():
        actual_hash = sha256_file(directory / name)
        if actual_hash != expected_hash:
            errors.append(f"hash_mismatch:{name}")

    fixture_dir = Path(config["paths"]["fixtures_dir"])
    fixture_cases = list(read_jsonl(fixture_dir / "reward_cases.jsonl"))
    fixture_classes = {row["expected_class"] for row in fixture_cases}
    required_fixture_classes = {
        "correct",
        "wrong_result",
        "execution_error",
        "unsafe_sql",
        "empty_output",
        "format_error",
    }
    missing_fixture_classes = sorted(required_fixture_classes - fixture_classes)
    if missing_fixture_classes:
        errors.append(f"reward_fixture_coverage:{missing_fixture_classes}")

    all_selected = canonical_train + canonical_validation + canonical_test
    report = {
        "data_version": version,
        "mode": mode,
        "valid": not errors,
        "errors": errors,
        "counts": expected_counts,
        "database_counts": {name: len(values) for name, values in db_sets.items()},
        "database_overlap": {
            "train_validation": sorted(db_sets["train"] & db_sets["validation"]),
            "train_test": sorted(db_sets["train"] & db_sets["test"]),
            "validation_test": sorted(db_sets["validation"] & db_sets["test"]),
        },
        "difficulty": count_values(all_selected, "difficulty"),
        "empty_result_counts": {
            name: sum(row["empty_result"] for row in rows)
            for name, rows in split_rows.items()
        },
        "token_length": {
            "prompt_p50": percentile([row["prompt_tokens"] for row in all_selected], 0.50),
            "prompt_p95": percentile([row["prompt_tokens"] for row in all_selected], 0.95),
            "prompt_max": max(row["prompt_tokens"] for row in all_selected),
            "sft_p50": percentile([row["sft_sequence_tokens"] for row in all_selected], 0.50),
            "sft_p95": percentile([row["sft_sequence_tokens"] for row in all_selected], 0.95),
            "sft_max": max(row["sft_sequence_tokens"] for row in all_selected),
        },
        "grpo": {
            "samples": len(grpo_train),
            "sft_overlap": len({row["id"] for row in grpo_train} & id_sets["train"]),
            "empty_gold_results": sum(
                row["reward_context"]["gold_result"] == [] for row in grpo_train
            ),
        },
        "reward_fixtures": {
            "cases": len(fixture_cases),
            "classes": sorted(fixture_classes),
        },
        "file_sha256": manifest["file_sha256"],
        "view_counts": {
            "sft_train": len(sft_train),
            "sft_validation": len(sft_validation),
            "grpo_train": len(grpo_train),
            "eval_validation": len(eval_validation),
            "eval_test": len(eval_test),
        },
    }
    return report, errors


def render_markdown(config: dict[str, Any], validation: dict[str, Any]) -> str:
    source = load_json(config["paths"]["source_manifest"])
    execution = load_json(config["paths"]["execution_summary"])
    clean = load_json(config["paths"]["clean_summary"])
    overlap = validation["database_overlap"]
    tokens = validation["token_length"]
    difficulty = validation["difficulty"]
    hashes = validation["file_sha256"]
    failure_lines = execution["failure_codes"] or {"none": 0}
    rejection_lines = clean["rejection_codes"] or {"none": 0}
    lines = [
        "# 阶段 1 数据报告",
        "",
        f"> 状态：{'已验收' if validation['valid'] else '未验收'}；数据版本：`{validation['data_version']}`。",
        "",
        "## 1. 来源与原始资产",
        "",
        f"- 来源：{source['source']} / {source['source_version']}；",
        f"- Train：{source['counts']['train_samples']} 条，Dev：{source['counts']['dev_samples']} 条；",
        f"- Schema：{source['counts']['table_entries']} 个，SQLite：{source['counts']['database_files']} 个；",
        f"- 官方 train/dev 数据库交集：{len(source['official_train_dev_db_overlap'])}；",
        f"- 许可记录：{source['license_note']}",
        "",
        "原始文件 SHA-256：",
        "",
        "| 文件 | SHA-256 |",
        "|---|---|",
    ]
    for info in source["files"].values():
        lines.append(f"| `{info['path']}` | `{info['sha256']}` |")
    lines.extend(
        [
            "",
            "## 2. Gold SQL 执行与清洗",
            "",
            f"- 输入：{execution['input_records']}；执行成功：{execution['successful_records']}；失败：{execution['failed_records']}；",
            f"- 执行成功率：{execution['success_rate'] * 100:.2f}%；成功记录中的空结果：{execution['empty_result_records']}；",
            f"- Token/去重后保留：{clean['kept_records']}；拒绝：{clean['rejected_records']}；",
            f"- 重复问题组：{clean['duplicate_question_groups']}；重复 SQL 组：{clean['duplicate_sql_groups']}。",
            "",
            "执行失败类型：`" + str(failure_lines) + "`",
            "",
            "清洗拒绝类型：`" + str(rejection_lines) + "`",
            "",
            "所有进入 processed 的 Gold SQL 都已在对应 SQLite 数据库只读执行成功；失败记录保存在 `data/interim/audit/`。",
            "",
            "## 3. Smoke 划分与派生视图",
            "",
            "| 资产 | 数量 |",
            "|---|---:|",
        ]
    )
    for name, count in validation["view_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            f"数据库数量：train={validation['database_counts']['train']}，validation={validation['database_counts']['validation']}，test={validation['database_counts']['test']}。",
            "",
            f"数据库泄漏：train/validation={len(overlap['train_validation'])}，train/test={len(overlap['train_test'])}，validation/test={len(overlap['validation_test'])}。",
            "",
            f"GRPO 样本：{validation['grpo']['samples']}；与 SFT train 重叠：{validation['grpo']['sft_overlap']}；空 Gold Result：{validation['grpo']['empty_gold_results']}。",
            "",
            "## 4. 长度与分布",
            "",
            f"- Prompt tokens：P50={tokens['prompt_p50']}，P95={tokens['prompt_p95']}，Max={tokens['prompt_max']}；",
            f"- SFT sequence tokens：P50={tokens['sft_p50']}，P95={tokens['sft_p95']}，Max={tokens['sft_max']}；",
            f"- 难度分布（启发式 v1）：`{difficulty}`；",
            f"- 最大序列长度：{config['tokenizer']['max_sequence_length']}；GRPO 预留 completion：{config['tokenizer']['grpo_max_new_tokens']} tokens。",
            "",
            "## 5. Reward 测试资产",
            "",
            f"已生成 {validation['reward_fixtures']['cases']} 个案例，覆盖：`{validation['reward_fixtures']['classes']}`。这些 fixture 不参与训练。",
            "",
            "## 6. 冻结哈希",
            "",
            "| 文件 | SHA-256 |",
            "|---|---|",
        ]
    )
    for name, digest in hashes.items():
        lines.append(f"| `{name}` | `{digest}` |")
    lines.extend(
        [
            "",
            "## 7. 验收",
            "",
            f"- [{'x' if validation['valid'] else ' '}] 数据来源、许可说明和哈希已记录；",
            f"- [{'x' if execution['successful_records'] + execution['failed_records'] == execution['input_records'] else ' '}] 所有原始 Gold SQL 均有执行或拒绝记录；",
            f"- [{'x' if not any(overlap.values()) else ' '}] `db_id` 跨 split 泄漏为 0；",
            f"- [{'x' if validation['grpo']['empty_gold_results'] == 0 else ' '}] GRPO smoke 不含空 Gold Result；",
            f"- [{'x' if validation['valid'] else ' '}] SFT/GRPO/Eval 视图、长度和哈希验证通过；",
            "",
            "阶段 1 只冻结 smoke 数据。正式 v1 必须由相同配置与脚本重新派生，不得手工扩充 processed 文件。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and freeze a processed dataset version.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    validation, errors = validate_dataset(config, args.mode)
    version = config["split"][args.mode]["version"]
    output_dir = Path(config["paths"]["processed_root"]) / version
    write_json(output_dir / "validation_report.json", validation)
    report_path = Path(config["paths"]["data_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(config, validation), encoding="utf-8", newline="\n")
    print(f"Validation errors: {len(errors)}")
    for error in errors[:20]:
        print(f"  - {error}")
    print(f"Wrote: {report_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
