"""One-sample SFT-policy generation followed by execution-based reward scoring."""
from __future__ import annotations

import argparse
import json
import os
import random
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel

from src.data.common import read_jsonl, sha256_file
from src.grpo.rewards import score_candidate
from src.inference.generate import Generator, verify_model
from src.utils.config import load_yaml, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = load_yaml(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "model_reward_smoke.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one device")
    if cfg["budget"]["external_downloads"]:
        raise ValueError("Stage 6 smoke must be offline")
    if shutil.disk_usage(".").free < float(cfg["budget"]["min_free_disk_gib"]) * 2**30:
        raise RuntimeError("Insufficient disk space")

    gpu_before = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info(0)[0] < float(cfg["budget"]["min_free_gpu_gib"]) * 2**30:
        raise RuntimeError("Insufficient free GPU memory")
    torch.cuda.reset_peak_memory_stats(0)

    manifest = json.loads(Path(cfg["dataset_manifest"]).read_text(encoding="utf-8"))
    dataset = Path(cfg["dataset"])
    if sha256_file(dataset) != manifest["file_sha256"][dataset.name]:
        raise ValueError("Frozen GRPO data hash mismatch")
    rows = list(read_jsonl(dataset))
    if not rows or rows[0]["id"] != manifest["sample_ids"]["grpo"][0]:
        raise ValueError("Frozen GRPO sample order mismatch")
    if int(cfg["budget"]["max_samples"]) != 1:
        raise ValueError("Stage 6 quick smoke is fixed to one sample")
    row = rows[0]

    model_cfg = load_yaml(cfg["model_config"])
    source = verify_model(model_cfg)
    adapter = Path(cfg["adapter_path"])
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Stage 5 best adapter is missing")
    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    torch.cuda.manual_seed_all(int(cfg["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    generator = Generator(model_cfg, cfg["generation"], source)
    generator.model = PeftModel.from_pretrained(generator.model, adapter, is_trainable=False).eval()
    prediction = generator.generate(row["prompt"])
    reward = score_candidate(prediction["raw_output"], row["reward_context"], cfg)
    if prediction["truncated"]:
        raise RuntimeError("Smoke output was truncated")
    if time.perf_counter() - started > float(cfg["budget"]["max_seconds"]):
        raise TimeoutError("Stage 6 smoke exceeded its wall-clock budget")
    if torch.cuda.mem_get_info(0)[0] < float(cfg["budget"]["gpu_safety_margin_gib"]) * 2**30:
        raise RuntimeError("GPU safety margin exhausted")

    report = {
        "status": "passed",
        "evidence_state": "executed",
        "command": " ".join(sys.argv),
        "sample_id": row["id"],
        "model": model_cfg["model"],
        "adapter_path": adapter.as_posix(),
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "dataset": dataset.as_posix(),
        "dataset_sha256": sha256_file(dataset),
        "prompt_contains_reward_context": False,
        "prediction": prediction,
        "reward": reward,
        "runtime": {
            "total_seconds": time.perf_counter() - started,
            "model_load_seconds": generator.load_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 2**30,
            "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "gpu_before": gpu_before.strip().splitlines(),
        },
    }
    write_json(result_path, report)
    print(json.dumps({
        "status": report["status"],
        "sample_id": report["sample_id"],
        "raw_output": prediction["raw_output"],
        "reward_class": reward["class"],
        "reward_total": reward["total"],
        "components": reward["components"],
        "runtime": report["runtime"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
