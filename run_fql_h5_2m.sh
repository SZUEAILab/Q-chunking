#!/bin/bash
# FQL H=5 + Post-hoc DS 对比实验（离线 1M → 在线 1M）
# baseline vs posthoc, 3 seeds each, 双卡并行
set -e

source .venv/bin/activate
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.05

RUN_GROUP="fql_h5_2m"
ENV="cube-triple-play-singletask-task2-v0"
COMMON="--run_group=$RUN_GROUP --env_name=$ENV \
  --eval_episodes=50 --video_episodes=0 --save_interval=1000000 \
  --offline_steps=1000000 --online_steps=1000000 --horizon_length=5"
LOG_DIR="logs/fql_h5_2m"

mkdir -p "$LOG_DIR"

run() {
  local tag="$1"; shift
  CUDA_VISIBLE_DEVICES="$gpu" taskset -c "$cores" \
    python main.py "$@" > "$LOG_DIR/$tag.log" 2>&1 &
}

echo "[$(date)] Starting FQL H=5 experiments"

# GPU 0: baseline (3 seeds)
gpu=0
cores=0-3   run h5_none_s0 $COMMON --seed=0;  g0_p1=$!
cores=4-7   run h5_none_s1 $COMMON --seed=1;  g0_p2=$!
cores=8-11  run h5_none_s2 $COMMON --seed=2;  g0_p3=$!

# GPU 1: posthoc (3 seeds)
gpu=1
cores=16-19 run h5_posthoc_s0 $COMMON --seed=0 --ds_mode=posthoc; g1_p1=$!
cores=20-23 run h5_posthoc_s1 $COMMON --seed=1 --ds_mode=posthoc; g1_p2=$!
cores=24-27 run h5_posthoc_s2 $COMMON --seed=2 --ds_mode=posthoc; g1_p3=$!

wait $g0_p1 $g0_p2 $g0_p3 $g1_p1 $g1_p2 $g1_p3
echo "[$(date)] All done"
