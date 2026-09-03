#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-configs/a5000.yaml}"
seed="${2:-42}"

fedicl-mqa train --config "$config_path" --mode local --seed "$seed" --resume auto
fedicl-mqa train --config "$config_path" --mode federated --seed "$seed" --resume auto

for round_index in 4 6 8; do
  fedicl-mqa evaluate \
    --config "$config_path" \
    --arm F0 \
    --seed "$seed" \
    --split validation \
    --round "$round_index"
done
