#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-configs/a5000.yaml}"
shift || true
if (($#)); then
  seeds=("$@")
else
  seeds=(42 43 44)
fi

for seed in "${seeds[@]}"; do
  bash scripts/train_seed.sh "$config_path" "$seed"
done

fedicl-mqa select-round --config "$config_path"
fedicl-mqa train --config "$config_path" --mode centralized --all-seeds --resume auto
