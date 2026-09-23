#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

cd "$project_root"
"$python_bin" -m src.model.inspect_architecture \
  --config configs/model.yaml \
  --json-out outputs/stage2/architecture.json \
  --markdown-out docs/architecture_notes.md
