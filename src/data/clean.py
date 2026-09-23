from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.common import build_messages, normalized_sql, read_jsonl, readonly_sql_reason, write_jsonl
from src.utils.config import load_yaml, write_json


def load_tokenizer(config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    tokenizer_config = config["tokenizer"]
    local_path = tokenizer_config.get("local_path")
    source = Path(local_path) if local_path else tokenizer_config["model_id"]
    if local_path and not source.is_dir():
        raise FileNotFoundError(f"Configured local tokenizer path does not exist: {source}")
    return AutoTokenizer.from_pretrained(
        source,
        local_files_only=bool(tokenizer_config["local_files_only"]),
        trust_remote_code=False,
        **({} if local_path else {"revision": tokenizer_config["revision"]}),
    )


def token_count(tokenizer: Any, messages: list[dict[str, str]], *, generation_prompt: bool) -> int:
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation_prompt,
    )
    if hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    if hasattr(tokens, "shape"):
        return int(tokens.shape[-1])
    if tokens and isinstance(tokens[0], list):
        return len(tokens[0])
    return len(tokens)


def clean_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tokenizer = load_tokenizer(config)
    tokenizer_config = config["tokenizer"]
    max_length = int(tokenizer_config["max_sequence_length"])
    grpo_new_tokens = int(tokenizer_config["grpo_max_new_tokens"])
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_codes: Counter[str] = Counter()
    seen_exact: set[tuple[str, str, str]] = set()
    question_counts: Counter[tuple[str, str]] = Counter()
    sql_counts: Counter[tuple[str, str]] = Counter()

    records = list(read_jsonl(config["paths"]["canonical_executed"]))
    for index, record in enumerate(records, start=1):
        reason = readonly_sql_reason(record["gold_sql"])
        if reason:
            code = f"unsafe_sql:{reason}"
        else:
            exact_key = (
                record["db_id"],
                " ".join(record["question"].split()).casefold(),
                normalized_sql(record["gold_sql"]).casefold(),
            )
            if exact_key in seen_exact:
                code = "exact_duplicate"
            else:
                seen_exact.add(exact_key)
                messages = build_messages(config, record["schema"], record["question"])
                prompt_tokens = token_count(tokenizer, messages, generation_prompt=True)
                sft_messages = messages + [
                    {"role": "assistant", "content": record["gold_sql"]}
                ]
                sft_tokens = token_count(tokenizer, sft_messages, generation_prompt=False)
                if sft_tokens > max_length:
                    code = "sft_sequence_too_long"
                elif prompt_tokens + grpo_new_tokens > max_length:
                    code = "grpo_sequence_too_long"
                else:
                    enriched = dict(record)
                    enriched.update(
                        {
                            "prompt_tokens": prompt_tokens,
                            "sft_sequence_tokens": sft_tokens,
                            "grpo_max_sequence_tokens": prompt_tokens + grpo_new_tokens,
                            "tokenizer_id": tokenizer_config["model_id"],
                            "max_sequence_length": max_length,
                        }
                    )
                    kept.append(enriched)
                    question_counts[(record["db_id"], exact_key[1])] += 1
                    sql_counts[(record["db_id"], exact_key[2])] += 1
                    continue

        rejection_codes[code] += 1
        rejected.append(
            {
                "id": record["id"],
                "db_id": record["db_id"],
                "source_split": record["source_split"],
                "reason_code": code,
            }
        )
        if index % 1000 == 0:
            print(f"Token/clean validation: {index}/{len(records)}")

    summary = {
        "input_records": len(records),
        "kept_records": len(kept),
        "rejected_records": len(rejected),
        "rejection_codes": dict(sorted(rejection_codes.items())),
        "kept_empty_results": sum(row["empty_result"] for row in kept),
        "duplicate_question_groups": sum(value > 1 for value in question_counts.values()),
        "duplicate_sql_groups": sum(value > 1 for value in sql_counts.values()),
        "max_prompt_tokens": max((row["prompt_tokens"] for row in kept), default=0),
        "max_sft_sequence_tokens": max(
            (row["sft_sequence_tokens"] for row in kept), default=0
        ),
    }
    return kept, rejected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean, deduplicate and tokenize canonical rows.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    kept, rejected, summary = clean_records(config)
    write_jsonl(config["paths"]["canonical_clean"], kept)
    write_jsonl(config["paths"]["clean_rejections"], rejected)
    write_json(config["paths"]["clean_summary"], summary)
    print(
        f"Clean complete: {summary['kept_records']} kept, "
        f"{summary['rejected_records']} rejected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
