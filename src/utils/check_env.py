from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from src.utils.config import write_json


TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "peft",
    "trl",
    "datasets",
    "PyYAML",
    "psutil",
    "pytest",
)


def _run(command: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _torch_probe() -> dict[str, Any]:
    probe = r"""
import json
import torch
payload = {
    "version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "compiled_cuda": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
    "devices": [],
}
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    payload["devices"].append({
        "index": index,
        "name": props.name,
        "total_memory_bytes": props.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    })
print(json.dumps(payload))
"""
    result = _run([sys.executable, "-c", probe], timeout=30)
    payload: dict[str, Any] = {"probe": result}
    if result["returncode"] == 0:
        try:
            payload.update(json.loads(result["stdout"].splitlines()[-1]))
        except (json.JSONDecodeError, IndexError) as exc:
            payload["parse_error"] = str(exc)
    return payload


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "error": "nvidia-smi not found"}
    query = (
        "index,name,driver_version,memory.total,memory.used,memory.free,"
        "temperature.gpu,power.draw,power.limit"
    )
    result = _run(
        [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        timeout=20,
    )
    devices: list[dict[str, Any]] = []
    if result["returncode"] == 0:
        for row in result["stdout"].splitlines():
            values = [value.strip() for value in row.split(",")]
            if len(values) != 9:
                continue
            devices.append(
                dict(
                    zip(
                        (
                            "index",
                            "name",
                            "driver_version",
                            "memory_total_mib",
                            "memory_used_mib",
                            "memory_free_mib",
                            "temperature_c",
                            "power_draw_w",
                            "power_limit_w",
                        ),
                        values,
                    )
                )
            )
    return {"available": result["returncode"] == 0, "devices": devices, "probe": result}


def collect_environment() -> dict[str, Any]:
    virtual_memory = psutil.virtual_memory()
    disk_entries = []
    seen: set[str] = set()
    for partition in psutil.disk_partitions(all=False):
        if partition.mountpoint in seen:
            continue
        seen.add(partition.mountpoint)
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except OSError:
            continue
        disk_entries.append(
            {
                "mountpoint": partition.mountpoint,
                "filesystem": partition.fstype,
                "total_gib": round(usage.total / 2**30, 2),
                "free_gib": round(usage.free / 2**30, 2),
            }
        )

    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
        },
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_total_gib": round(virtual_memory.total / 2**30, 2),
            "memory_available_gib": round(virtual_memory.available / 2**30, 2),
            "disks": disk_entries,
        },
        "packages": _package_versions(),
        "torch": _torch_probe(),
        "nvidia_smi": _nvidia_smi(),
        "nvcc": _run(["nvcc", "--version"]) if shutil.which("nvcc") else {"available": False},
    }


def _print_summary(report: dict[str, Any]) -> None:
    python_info = report["python"]
    system_info = report["system"]
    torch_info = report["torch"]
    print(f"Python: {python_info['version']} ({python_info['executable']})")
    print(f"OS: {system_info['platform']}")
    print(
        f"CPU: {system_info['physical_cpu_count']} cores / "
        f"{system_info['logical_cpu_count']} logical"
    )
    print(
        f"RAM: {system_info['memory_available_gib']:.2f} GiB available / "
        f"{system_info['memory_total_gib']:.2f} GiB total"
    )
    print("Packages:")
    for name, version in report["packages"].items():
        print(f"  {name}: {version or 'NOT INSTALLED'}")
    if torch_info.get("version"):
        print(
            f"PyTorch: {torch_info['version']}, compiled CUDA: {torch_info.get('compiled_cuda')}, "
            f"CUDA available: {torch_info.get('cuda_available')}"
        )
        for device in torch_info.get("devices", []):
            print(
                f"  GPU {device['index']}: {device['name']}, "
                f"{device['total_memory_bytes'] / 2**30:.2f} GiB, "
                f"CC {'.'.join(map(str, device['compute_capability']))}, "
                f"BF16={device['bf16_supported']}"
            )
    else:
        probe = torch_info.get("probe", {})
        print(f"PyTorch probe failed: {probe.get('stderr') or probe.get('stdout')}")
    if not report["nvcc"].get("available", report["nvcc"].get("returncode") == 0):
        print("NVCC: not installed (not required when using prebuilt PyTorch wheels)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the local ML environment.")
    parser.add_argument("--json-out", type=Path, help="Write the full report as JSON.")
    parser.add_argument("--strict", action="store_true", help="Fail if CUDA PyTorch is unavailable.")
    args = parser.parse_args()

    report = collect_environment()
    _print_summary(report)
    if args.json_out:
        write_json(args.json_out, report)
        print(f"Wrote: {args.json_out}")

    cuda_ok = bool(report["torch"].get("cuda_available"))
    packages_ok = all(report["packages"].get(name) for name in ("torch", "transformers", "PyYAML"))
    if args.strict and not (cuda_ok and packages_ok):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

