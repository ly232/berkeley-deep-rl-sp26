#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p run_logs

max_jobs="${MAX_JOBS:-100}"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$max_jobs" ]; do
    sleep 2
  done
}

for lambda in 0.95 0.98 1; do
  for lr in 0.0001 0.0003 0.001 0.003; do
    for batch_size in 1000 5000; do
      exp_name="pendulum_lambda${lambda}_lr${lr}_na_b${batch_size}"
      log_path="run_logs/${exp_name}.log"

      wait_for_slot
      uv run src/scripts/run.py \
        --env_name InvertedPendulum-v4 \
        -n 100 \
        -b "$batch_size" \
        -eb 1000 \
        -lr "$lr" \
        --use_reward_to_go \
        --use_baseline \
        --gae_lambda "$lambda" \
        -na \
        --exp_name "$exp_name" \
        > "$log_path" 2>&1 &
    done
  done
done

wait
