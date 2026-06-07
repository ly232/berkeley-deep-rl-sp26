#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for lambda in 0 0.95 0.98 0.99 1; do
  uv run src/scripts/run.py \
    --env_name LunarLander-v2 \
    --ep_len 1000 \
    --discount 0.99 \
    -n 200 \
    -b 2000 \
    -eb 2000 \
    -l 3 \
    -s 128 \
    -lr 0.001 \
    --use_reward_to_go \
    --use_baseline \
    --gae_lambda "$lambda" \
    --exp_name "lunar_lander_lambda${lambda}" \
    > "run_logs/lunar_lander_lambda${lambda}.log" 2>&1 &
done

wait
