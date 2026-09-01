#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
torch_version="${TORCH_VERSION:-2.8.0}"
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

"$python_bin" -m pip install --upgrade pip setuptools wheel
"$python_bin" -m pip install "torch==$torch_version" --index-url "$torch_index_url"
"$python_bin" -m pip install -r "$project_root/requirements.txt"
echo "Dependencies installed into active environment: $($python_bin -c 'import sys; print(sys.executable)')"
