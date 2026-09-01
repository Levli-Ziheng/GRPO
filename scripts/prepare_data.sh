#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
mode="smoke"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="${2:?--mode requires smoke or full}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
  echo "--mode must be smoke or full" >&2
  exit 2
fi

cd "$project_root"
"$python_bin" -m src.data.download --config configs/data.yaml
"$python_bin" -m src.data.convert --config configs/data.yaml --mode "$mode"
"$python_bin" -m src.data.compute_gold_results --config configs/data.yaml
"$python_bin" -m src.data.clean --config configs/data.yaml
"$python_bin" -m src.data.split --config configs/data.yaml --mode "$mode"
"$python_bin" -m src.data.reward_fixtures --config configs/data.yaml
"$python_bin" -m src.data.validate --config configs/data.yaml --mode "$mode"

echo "Stage-1 data preparation completed for mode=$mode."
