"""Stage-3 sequential, offline generation with explicit experiment provenance."""
from __future__ import annotations
import argparse
import copy
import resource
import signal
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import yaml
from src.data.common import build_messages, read_jsonl, sha256_file
from src.utils.config import load_yaml, write_json
from src.model.load import resolve_model_source, _torch_dtype

def load_tasks(config, data_config):
    path = Path(config["dataset"])
    manifest = json.loads(Path(config["dataset_manifest"]).read_text())
    if sha256_file(path) != manifest["file_sha256"][path.name]:
        raise ValueError("Frozen test hash mismatch")
    rows = list(read_jsonl(path))
    if len(rows) != config["expected_samples"] or [r["id"] for r in rows] != manifest["sample_ids"]["test"]:
        raise ValueError("Frozen test IDs/count mismatch")
    eval_rows = {r["id"]: r for r in read_jsonl(path.with_name("eval_test.jsonl"))}
    checked = set()
    for row in rows:
        if build_messages(data_config, row["schema"], row["question"]) != eval_rows[row["id"]]["prompt"]:
            raise ValueError("Prompt differs from frozen evaluation prompt")
        db = row["db_path"]
        if db not in checked:
            if sha256_file(db) != row["db_sha256"]:
                raise ValueError(f"Database hash drift: {db}")
            checked.add(db)
    return rows

def verify_model(model_config):
    source, local = resolve_model_source(model_config["model"])
    if not local or not model_config["model"]["local_files_only"]:
        raise ValueError("Stage 3 requires the existing offline model")
    manifest = json.loads((Path(source) / "download_manifest.json").read_text())
    if manifest["revision"] != model_config["model"]["revision"] or manifest["model_id"] != model_config["model"]["id"]:
        raise ValueError("Model revision mismatch")
    for name, expected in manifest["files"].items():
        if sha256_file(Path(source) / name) != expected:
            raise ValueError(f"Local model hash mismatch: {name}")
    return source

class Generator:
    def __init__(self, model_config, generation_config, source):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
        self.torch = torch
        self.config = generation_config
        runtime, quant = model_config["runtime"], model_config["quantization"]
        self.device = torch.device(runtime["device"])
        torch.cuda.init()
        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        kwargs = {"local_files_only": True, "trust_remote_code": False,
                  "dtype": _torch_dtype(torch, runtime["dtype"]), "device_map": {"": self.device.index or 0},
                  "attn_implementation": "sdpa"}
        if runtime["quantization"] == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type=quant["quant_type"],
                bnb_4bit_compute_dtype=_torch_dtype(torch, quant["compute_dtype"]),
                bnb_4bit_use_double_quant=quant["use_double_quant"])
        elif runtime["quantization"] != "none":
            raise ValueError("Unsupported quantization")
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(source, **kwargs).eval()
        torch.cuda.synchronize(self.device)
        self.load_seconds = time.perf_counter() - started
        # Do not inherit sampling knobs from the checkpoint's generation_config.json.
        self.decoding = GenerationConfig(do_sample=False, num_beams=1, use_cache=True, temperature=None, top_p=None, top_k=None,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)

    def generate(self, messages):
        torch = self.torch
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt").to(self.device)
        count = inputs["input_ids"].shape[-1]
        allowance = min(self.config["max_new_tokens"], self.config["max_sequence_length"] - count)
        if allowance <= 0:
            raise ValueError(f"Prompt exceeds context budget: {count}")
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            decoding = copy.deepcopy(self.decoding)
            decoding.max_new_tokens = allowance
            output = self.model.generate(**inputs, generation_config=decoding)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        tokens = output[0, count:].tolist()
        eos = self.tokenizer.eos_token_id
        return {"prompt": self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                "raw_output": self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                "input_tokens": count, "output_tokens": len(tokens), "max_new_tokens_effective": allowance,
                "latency_seconds": elapsed, "truncated": len(tokens) >= allowance and tokens[-1] != eos}

def run(args):
    import torch
    from src.evaluation.evaluate import evaluate_record, write_evaluation
    cfg = load_yaml(args.config)
    model_cfg = load_yaml(cfg["model_config"])
    data_cfg = load_yaml(cfg["data_config"])
    if cfg["generation"]["do_sample"] or cfg["generation"]["num_beams"] != 1:
        raise ValueError("This baseline requires greedy decoding")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one device")
    rows = load_tasks(cfg, data_cfg)
    if args.mode == "single":
        rows = rows[:1]
    if args.question:
        if args.mode != "single" or not args.schema_file:
            raise ValueError("--question requires --mode single and --schema-file (schema is mandatory)")
        rows = [{"id": "custom", "question": args.question, "schema": Path(args.schema_file).read_text(),
                 "difficulty": "custom", "db_id": None}]
    if len(rows) > cfg["budget"]["max_samples"]:
        raise ValueError("Sample budget exceeded")
    out = Path(args.output_dir) if args.output_dir else Path("outputs/baseline") / (
        "base-" + args.mode + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    command = "CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"] + " " + shlex.join([sys.executable, "-m", sys.modules["__main__"].__spec__.name, *sys.argv[1:]])
    write_json(out / "status.json", {"status": "running", "command": command})
    resolved = {"baseline": cfg, "model": model_cfg, "data": data_cfg,
                "mode": args.mode, "stage": args.stage, "sample_ids": [r["id"] for r in rows],
                "command": command, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                "attention_implementation": "sdpa", "decoding": "greedy; default repetition penalty 1; cache enabled"}
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False))
    try:
        source = verify_model(model_cfg)
        if shutil.disk_usage(".").free < cfg["budget"]["min_free_disk_gib"] * 2**30:
            raise RuntimeError("Insufficient disk")
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True)
        (out / "gpu_before.csv").write_text(gpu)
        if torch.cuda.mem_get_info(0)[0] < cfg["budget"]["min_free_gpu_gib"] * 2**30:
            raise RuntimeError("Insufficient free GPU memory")
        subprocess.run([sys.executable, "-m", "src.utils.check_env", "--json-out", str(out / "environment.json")],
                       check=True, stdout=(out / "environment.log").open("w"))
        artifacts = [args.config, cfg["model_config"], cfg["data_config"], cfg["dataset_manifest"],
                     cfg["dataset"], str(Path(source) / "download_manifest.json"), str(out / "environment.json")]
        context_cmd = [sys.executable, ".agents/skills/ai-experiment/scripts/capture_context.py", "--repo-root", ".",
                       "--output", str(out / "run_context.json"), "--experiment-id", out.name, "--command", command]
        for artifact in artifacts:
            context_cmd += ["--artifact", artifact]
        subprocess.run(context_cmd, check=True)
        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        generator = Generator(model_cfg, cfg["generation"], source)
        predictions = []
        with (out / "predictions.jsonl").open("x", encoding="utf-8") as handle:
            for idx, row in enumerate(rows, 1):
                if time.perf_counter() - start >= cfg["budget"]["max_seconds"]:
                    raise TimeoutError("Experiment time budget exhausted")
                if torch.cuda.mem_get_info(0)[0] < model_cfg["runtime"]["gpu_safety_margin_gib"] * 2**30:
                    raise RuntimeError("GPU safety margin exhausted")
                # Only question + schema enter the policy model, never evaluation fields.
                prediction = generator.generate(build_messages(data_cfg, row["schema"], row["question"]))
                prediction.update({"id": row["id"], "stage": args.stage, "model": model_cfg["model"]["id"],
                                   "model_revision": model_cfg["model"]["revision"]})
                if row["id"] != "custom":
                    prediction = evaluate_record(prediction, row, cfg["evaluation"])
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                handle.flush()
                predictions.append(prediction)
                print(f"{idx}/{len(rows)} {row['id']} tokens={prediction['output_tokens']} correct={prediction.get('execution_correct')} truncated={prediction['truncated']}", flush=True)
                if prediction["truncated"]:
                    raise RuntimeError("Output truncated; preserve partial results and diagnose before expansion")
        if rows[0]["id"] != "custom":
            metrics = write_evaluation(out, predictions)
        else:
            metrics = {}
        runtime = {"peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                   "load_seconds": generator.load_seconds, "total_seconds": time.perf_counter() - start,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 2**30,
                   "gpu": torch.cuda.get_device_name(0), "sample_count": len(predictions)}
        write_json(out / "runtime.json", runtime)
        # Verify data/database invariants again after execution.
        load_tasks(cfg, data_cfg)
        write_json(out / "status.json", {"status": "completed", "evidence_state": "executed",
                   "completed_at_utc": datetime.now(timezone.utc).isoformat(), "command": command})
        print(json.dumps({"output_dir": str(out), "metrics": metrics, "runtime": runtime}, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        write_json(out / "status.json", {"status": "failed", "error": repr(exc), "command": command,
                                       "elapsed_seconds": time.perf_counter() - start})
        raise

def main():
    def terminate(signum, frame):
        raise TimeoutError("External wall-clock budget exhausted")
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--stage", choices=["base"], default="base")
    parser.add_argument("--mode", choices=["single", "batch"], default="single")
    parser.add_argument("--output-dir")
    parser.add_argument("--question")
    parser.add_argument("--schema-file")
    return run(parser.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
