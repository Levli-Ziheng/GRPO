#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python_bin="${PYTHON_BIN:-python3}"
mode=smoke
previous=""
for argument in "$@"; do
  if [[ "$previous" == "--mode" ]]; then mode="$argument"; fi
  case "$argument" in --mode=*) mode="${argument#--mode=}" ;; esac
  previous="$argument"
done
seconds=1800
if [[ "$mode" == "full" ]]; then seconds=7500; fi
timeout --signal=TERM "$seconds" "$python_bin" -u -m src.sft.train "$@"
