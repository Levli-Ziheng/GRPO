#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

cd "$project_root"
"$python_bin" -m src.model.load --config configs/model.yaml --smoke-test --json-out outputs/stage0/model_smoke.json
