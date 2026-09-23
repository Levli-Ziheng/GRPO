"""Explicit assistant-SQL labels; no template-dependent implicit loss mask."""
from __future__ import annotations
import random
from pathlib import Path
import torch
from src.data.common import read_jsonl, load_json, sha256_file, build_messages
from src.utils.config import load_yaml

def encode_example(row, tokenizer, max_length):
    messages = row["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected exactly system/user/assistant messages")
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat-template token prefix is not stable")
    if len(full) > max_length:
        raise ValueError(f"Refusing to truncate SQL: {row['id']} ({len(full)} tokens)")
    if row.get("prompt_tokens", len(prompt)) != len(prompt) or row.get("sequence_tokens", len(full)) != len(full):
        raise ValueError(f"Frozen token count drift: {row['id']}")
    try:
        eos_at = full.index(tokenizer.eos_token_id, len(prompt))
    except ValueError as exc:
        raise ValueError("Assistant completion lacks EOS") from exc
    if tokenizer.decode(full[len(prompt):eos_at], skip_special_tokens=False) != messages[-1]["content"]:
        raise ValueError("Token boundary does not decode to exact assistant SQL")
    # Preserve the full chat sequence, but mask header, input and trailing template newline.
    labels = [-100] * len(full)
    labels[len(prompt):eos_at+1] = full[len(prompt):eos_at+1]
    if eos_at <= len(prompt):
        raise ValueError("Empty SQL target")
    return {"input_ids": full, "attention_mask": [1]*len(full), "labels": labels}

class SQLDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.features = [encode_example(r, tokenizer, max_length) for r in rows]
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index]

class SQLCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    def __call__(self, features):
        width = max(len(f["input_ids"]) for f in features)
        batch = {k: [] for k in ("input_ids", "attention_mask", "labels")}
        for feature in features:
            n = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id]*n)
            batch["attention_mask"].append(feature["attention_mask"] + [0]*n)
            batch["labels"].append(feature["labels"] + [-100]*n)
        return {k: torch.tensor(v, dtype=torch.long) for k,v in batch.items()}

def load_frozen_rows(config):
    root = Path(config["dataset_root"])
    manifest = load_json(root/config["manifest_file"])
    for filename, expected in manifest["file_sha256"].items():
        if sha256_file(root/filename) != expected:
            raise ValueError(f"Frozen data hash drift: {filename}")
    train, validation = [list(read_jsonl(root/config[k])) for k in ("train_file","validation_file")]
    if len(train)>config["budget"]["max_train_samples"] or len(validation)>config["budget"]["max_validation_samples"]:
        raise ValueError("Data exceeds smoke budget")
    data_config = load_yaml(config["data_config"])
    for split, rows in (("train",train),("validation",validation)):
        canonical = {r["id"]:r for r in read_jsonl(root/f"canonical_{split}.jsonl")}
        if [r["id"] for r in rows] != manifest["sample_ids"][split]:
            raise ValueError("Frozen sample ID mismatch")
        for row in rows:
            task = canonical[row["id"]]
            expected = build_messages(data_config,task["schema"],task["question"])
            if row["messages"][:-1] != expected or row["messages"][-1] != {"role":"assistant","content":task["gold_sql"]}:
                raise ValueError("SFT prompt/SQL differs from canonical data")
    dbsets = [set(r["db_id"] for r in rows) for rows in (train,validation)]
    testdbs = set(manifest["database_ids"]["test"])
    if dbsets[0]&dbsets[1] or (dbsets[0]|dbsets[1])&testdbs:
        raise ValueError("Database split leakage")
    return train, validation

def mask_audit(dataset, tokenizer, seed, count):
    indices = random.Random(seed).sample(range(len(dataset)), min(count,len(dataset)))
    report=[]
    for i in indices:
        f=dataset[i]
        supervised = [x for x in f["labels"] if x != -100]
        report.append({"id":dataset.rows[i]["id"],"sequence_tokens":len(f["input_ids"]),
            "masked_tokens":f["labels"].count(-100),"supervised_tokens":len(supervised),
            "supervised_text":tokenizer.decode(supervised,skip_special_tokens=False),
            "expected_sql":dataset.rows[i]["messages"][-1]["content"],
            "input_ids":f["input_ids"],"labels":f["labels"]})
    return report
