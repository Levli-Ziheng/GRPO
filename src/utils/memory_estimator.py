from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.config import load_yaml, write_json


GIB = 2**30


def parameter_storage_gib(parameter_count: int, bits_per_parameter: float) -> float:
    return parameter_count * bits_per_parameter / 8 / GIB


def kv_cache_gib(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    sequence_length: int,
    batch_size: int = 1,
    bytes_per_element: int = 2,
) -> float:
    byte_count = (
        layers
        * 2
        * kv_heads
        * head_dim
        * sequence_length
        * batch_size
        * bytes_per_element
    )
    return byte_count / GIB


def estimate(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    architecture = config["architecture"]
    runtime = config["runtime"]
    parameter_count = int(model["parameter_count"])
    sequence_length = int(runtime["max_seq_length"])
    safety_margin = float(runtime["gpu_safety_margin_gib"])

    storage = {
        "fp32_gib": round(parameter_storage_gib(parameter_count, 32), 3),
        "fp16_gib": round(parameter_storage_gib(parameter_count, 16), 3),
        "int8_gib": round(parameter_storage_gib(parameter_count, 8), 3),
        "int4_raw_gib": round(parameter_storage_gib(parameter_count, 4), 3),
        "int4_with_20pct_overhead_gib": round(
            parameter_storage_gib(parameter_count, 4) * 1.2, 3
        ),
    }
    kv = kv_cache_gib(
        layers=int(architecture["num_hidden_layers"]),
        kv_heads=int(architecture["num_key_value_heads"]),
        head_dim=int(architecture["head_dim"]),
        sequence_length=sequence_length,
    )

    # These are conservative planning ranges, not substitutes for a real smoke test.
    fp16_inference = storage["fp16_gib"] + kv + 0.6
    int4_inference = storage["int4_with_20pct_overhead_gib"] + kv + 0.8
    # Training activations and rollout buffers dominate beyond parameter storage.
    # The ranges intentionally leave room for allocator fragmentation on a 24 GiB card.
    qlora_low = int4_inference + 4.0 + safety_margin
    qlora_high = int4_inference + 12.0 + safety_margin
    grpo_low = int4_inference * 2 + 8.0 + safety_margin
    grpo_high = int4_inference * 2 + 16.0 + safety_margin

    return {
        "model_id": model["id"],
        "parameter_count": parameter_count,
        "max_seq_length": sequence_length,
        "parameter_storage": storage,
        "kv_cache_fp16_gib": round(kv, 3),
        "planning_estimates": {
            "fp16_inference_gib": round(fp16_inference, 2),
            "int4_inference_gib": round(int4_inference, 2),
            "qlora_training_range_gib": [round(qlora_low, 2), round(qlora_high, 2)],
            "grpo_range_gib": [round(grpo_low, 2), round(grpo_high, 2)],
        },
        "assumptions": [
            "batch_size=1",
            "FP16 KV cache",
            "short 512-token sequence",
            "QLoRA activation checkpointing and gradient accumulation",
            "GRPO is highly implementation- and group-size-dependent",
            "Only a real smoke test can establish the final peak VRAM",
        ],
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"Model: {report['model_id']}")
    print(f"Parameters: {report['parameter_count']:,}")
    print("Parameter-only storage:")
    for key, value in report["parameter_storage"].items():
        print(f"  {key}: {value:.3f} GiB")
    print(f"KV cache at {report['max_seq_length']} tokens: {report['kv_cache_fp16_gib']:.3f} GiB")
    print("Conservative planning estimates:")
    for key, value in report["planning_estimates"].items():
        print(f"  {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate model memory requirements.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = estimate(load_yaml(args.config))
    _print_report(report)
    if args.json_out:
        write_json(args.json_out, report)
        print(f"Wrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
