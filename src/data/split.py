from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.data.common import build_messages, read_jsonl, sha256_file, write_jsonl
from src.utils.config import load_yaml, write_json


def balanced_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"Requested {count} rows from a pool of {len(rows)}")
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["difficulty"], row["db_id"])].append(row)
    keys = list(groups)
    rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)

    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        made_progress = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            break
        rng.shuffle(keys)
    if len(selected) != count:
        raise RuntimeError(f"Balanced sampling produced {len(selected)} of {count} rows")
    return sorted(selected, key=lambda row: row["id"])


def choose_validation_databases(
    train_rows: list[dict[str, Any]], count: int, seed: int
) -> list[str]:
    db_ids = sorted({row["db_id"] for row in train_rows})
    rng = random.Random(seed)
    rng.shuffle(db_ids)
    if len(db_ids) <= count:
        raise ValueError("Not enough training databases to create a held-out validation split")
    return sorted(db_ids[:count])


def canonical_view(row: dict[str, Any], split: str, version: str) -> dict[str, Any]:
    result = dict(row)
    result["split"] = split
    result["data_version"] = version
    return result


def sft_view(config: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    messages = build_messages(config, row["schema"], row["question"])
    messages.append({"role": "assistant", "content": row["gold_sql"]})
    return {
        "id": row["id"],
        "source_id": row["id"],
        "db_id": row["db_id"],
        "messages": messages,
        "prompt_tokens": row["prompt_tokens"],
        "sequence_tokens": row["sft_sequence_tokens"],
        "difficulty": row["difficulty"],
    }


def prompt_view(config: dict[str, Any], row: dict[str, Any]) -> list[dict[str, str]]:
    return build_messages(config, row["schema"], row["question"])


def grpo_view(config: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    execution = config["execution"]
    return {
        "id": row["id"],
        "source_id": row["id"],
        "db_id": row["db_id"],
        "prompt": prompt_view(config, row),
        "prompt_tokens": row["prompt_tokens"],
        "difficulty": row["difficulty"],
        "reward_context": {
            "db_path": row["db_path"],
            "db_sha256": row["db_sha256"],
            "gold_result": row["gold_result"],
            "gold_result_hash": row["gold_result_hash"],
            "result_compare_mode": row["result_compare_mode"],
            "order_sensitive": row["order_sensitive"],
            "float_tolerance": row["float_tolerance"],
            "timeout_seconds": float(execution["timeout_seconds"]),
            "max_result_rows": int(execution["max_result_rows"]),
        },
    }


def eval_view(config: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "source_id": row["id"],
        "db_id": row["db_id"],
        "prompt": prompt_view(config, row),
        "prompt_tokens": row["prompt_tokens"],
        "difficulty": row["difficulty"],
        "sql_features": row["sql_features"],
        "evaluation_context": {
            "db_path": row["db_path"],
            "db_sha256": row["db_sha256"],
            "gold_sql": row["gold_sql"],
            "gold_result": row["gold_result"],
            "gold_result_hash": row["gold_result_hash"],
            "result_compare_mode": row["result_compare_mode"],
            "order_sensitive": row["order_sensitive"],
            "float_tolerance": row["float_tolerance"],
        },
    }


def prepare_split(config: dict[str, Any], mode: str) -> dict[str, Any]:
    split_config = config["split"]
    size_config = split_config[mode]
    seed = int(split_config["seed"])
    version = size_config["version"]
    rows = list(read_jsonl(config["paths"]["canonical_clean"]))
    source_train = [row for row in rows if row["source_split"] == "train"]
    source_dev = [row for row in rows if row["source_split"] == "dev"]
    validation_db_ids = choose_validation_databases(
        source_train, int(split_config["validation_database_count"]), seed
    )
    validation_db_set = set(validation_db_ids)
    train_pool = [row for row in source_train if row["db_id"] not in validation_db_set]
    validation_pool = [row for row in source_train if row["db_id"] in validation_db_set]

    train_rows = balanced_sample(train_pool, int(size_config["sft_train"]), seed + 1)
    validation_rows = balanced_sample(
        validation_pool, int(size_config["validation"]), seed + 2
    )
    test_rows = balanced_sample(source_dev, int(size_config["test"]), seed + 3)
    grpo_pool = [row for row in train_rows if not row["empty_result"]]
    grpo_rows = balanced_sample(grpo_pool, int(size_config["grpo_train"]), seed + 4)

    canonical_train = [canonical_view(row, "train", version) for row in train_rows]
    canonical_validation = [
        canonical_view(row, "validation", version) for row in validation_rows
    ]
    canonical_test = [canonical_view(row, "test", version) for row in test_rows]
    output_dir = Path(config["paths"]["processed_root"]) / version
    outputs: dict[str, list[dict[str, Any]]] = {
        "canonical_train.jsonl": canonical_train,
        "canonical_validation.jsonl": canonical_validation,
        "canonical_test.jsonl": canonical_test,
        "sft_train.jsonl": [sft_view(config, row) for row in train_rows],
        "sft_validation.jsonl": [sft_view(config, row) for row in validation_rows],
        "grpo_train.jsonl": [grpo_view(config, row) for row in grpo_rows],
        "eval_validation.jsonl": [eval_view(config, row) for row in validation_rows],
        "eval_test.jsonl": [eval_view(config, row) for row in test_rows],
    }
    file_counts = {}
    for name, records in outputs.items():
        file_counts[name] = write_jsonl(output_dir / name, records)

    train_db_ids = sorted({row["db_id"] for row in train_rows})
    test_db_ids = sorted({row["db_id"] for row in test_rows})
    manifest = {
        "source": config["source"]["name"],
        "mode": mode,
        "data_version": version,
        "seed": seed,
        "counts": file_counts,
        "database_counts": {
            "train": len(train_db_ids),
            "validation": len(set(validation_db_ids)),
            "test": len(test_db_ids),
        },
        "database_ids": {
            "train": train_db_ids,
            "validation": validation_db_ids,
            "test": test_db_ids,
        },
        "sample_ids": {
            "train": [row["id"] for row in train_rows],
            "validation": [row["id"] for row in validation_rows],
            "test": [row["id"] for row in test_rows],
            "grpo": [row["id"] for row in grpo_rows],
        },
        "grpo_sft_overlap_count": len(
            {row["id"] for row in train_rows} & {row["id"] for row in grpo_rows}
        ),
        "grpo_empty_gold_results": sum(row["empty_result"] for row in grpo_rows),
    }
    manifest_path = output_dir / "dataset_manifest.json"
    write_json(manifest_path, manifest)
    file_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.glob("*.jsonl"))
    }
    manifest["file_sha256"] = file_hashes
    write_json(manifest_path, manifest)
    write_json(config["paths"]["split_manifest"], manifest)
    print(
        f"Prepared {version}: SFT={len(train_rows)}, GRPO={len(grpo_rows)}, "
        f"validation={len(validation_rows)}, test={len(test_rows)}"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create leak-free CSpider dataset views.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    prepare_split(config, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
