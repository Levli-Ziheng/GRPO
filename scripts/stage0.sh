#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$project_root"
bash scripts/bootstrap.sh
bash scripts/check_env.sh
bash scripts/smoke_model.sh

echo "Stage-0 commands finished. Review outputs/stage0 and fill docs/compute_report.md."
