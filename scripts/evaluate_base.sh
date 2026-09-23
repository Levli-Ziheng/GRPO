#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python_bin="${PYTHON_BIN:-python3}"
timeout --signal=TERM 1800 "$python_bin" -u -m src.evaluation.evaluate --config configs/baseline.yaml --stage base "$@"
