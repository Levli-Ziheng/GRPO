from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from src.utils.config import load_yaml, write_json


def _torch_dtype(torch_module: Any, name: str) -> Any:
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def run_smoke_test(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_config = config["model"]
    runtime = config["runtime"]
    quantization = config["quantization"]
    smoke = config["smoke_test"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for the stage-0 model smoke test.")

    model_id = model_config["id"]
    dtype = _torch_dtype(torch, runtime["dtype"])
    quantization_mode = runtime["quantization"]
    cuda_device = torch.device(runtime["device"])
    if cuda_device.type != "cuda":
        raise ValueError(f"Stage-0 smoke test requires a CUDA device, got: {cuda_device}")
    cuda_index = cuda_device.index if cuda_device.index is not None else 0
    load_kwargs: dict[str, Any] = {
        "revision": model_config["revision"],
        "trust_remote_code": bool(model_config["trust_remote_code"]),
        "local_files_only": bool(model_config["local_files_only"]),
        "torch_dtype": dtype,
        "device_map": {"": cuda_index},
    }
    if quantization_mode == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization["quant_type"],
            bnb_4bit_compute_dtype=_torch_dtype(torch, quantization["compute_dtype"]),
            bnb_4bit_use_double_quant=bool(quantization["use_double_quant"]),
        )
    elif quantization_mode != "none":
        raise ValueError(f"Unsupported quantization mode: {quantization_mode}")

    # Some PyTorch builds reject peak-memory operations before the CUDA
    # context is initialized. Initialize and select the configured GPU first.
    torch.cuda.init()
    torch.cuda.set_device(cuda_device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(cuda_device)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_config["revision"],
        trust_remote_code=bool(model_config["trust_remote_code"]),
        local_files_only=bool(model_config["local_files_only"]),
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    load_seconds = time.perf_counter() - started

    messages = [
        {"role": "system", "content": smoke["system_prompt"]},
        {"role": "user", "content": smoke["user_prompt"]},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generation_started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(smoke["max_new_tokens"]),
            do_sample=bool(smoke["do_sample"]),
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    new_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    report = {
        "model_id": model_id,
        "requested_revision": model_config["revision"],
        "resolved_commit": getattr(model.config, "_commit_hash", None),
        "dtype": runtime["dtype"],
        "quantization": quantization_mode,
        "device": str(model.device),
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "output_tokens": int(new_tokens.shape[-1]),
        "generated_text": generated_text,
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(cuda_device) / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(cuda_device) / 2**30, 3),
        "gpu_name": torch.cuda.get_device_name(cuda_device),
        "success": bool(generated_text),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Load the locked model and run a minimal generation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true", required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = run_smoke_test(load_yaml(args.config))
    print(f"Model: {report['model_id']}")
    print(f"Device: {report['device']} ({report['gpu_name']})")
    print(f"Precision: {report['dtype']}, quantization: {report['quantization']}")
    print(f"Generated: {report['generated_text']}")
    print(
        f"Load: {report['load_seconds']:.3f}s, generation: {report['generation_seconds']:.3f}s, "
        f"peak allocated: {report['peak_allocated_gib']:.3f} GiB, "
        f"peak reserved: {report['peak_reserved_gib']:.3f} GiB"
    )
    if args.json_out:
        write_json(args.json_out, report)
        print(f"Wrote: {args.json_out}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
