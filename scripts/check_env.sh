#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

cd "$project_root"
"$python_bin" -m src.utils.check_env --strict --json-out outputs/stage0/environment.json
"$python_bin" -m src.utils.memory_estimator --config configs/model.yaml --json-out outputs/stage0/memory_estimate.json
