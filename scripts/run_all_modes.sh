#!/usr/bin/env bash
set -euo pipefail

# Batch runner: iterate models and cache modes via sample.sh

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_ROOT/data/prompt_eval_1000_v2.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/results_batch}"
# 模型列表：包含 flux-dev，如需新增可在此数组追加
MODELS=("flux-dev")
MODES=("Taylor" "Taylor-Scaled" "HiCache" "original" "ToCa" "Delta" "ClusCa" "Hi-ClusCa")
INTERVAL="${INTERVAL:-5}"
MAX_ORDER="${MAX_ORDER:-2}"
HICACHE_SCALE="${HICACHE_SCALE:-0.7}"
FIRST_ENHANCE="${FIRST_ENHANCE:-3}"
NUM_STEPS_DEV="${NUM_STEPS_DEV:-50}"
NUM_STEPS_SCHNELL="${NUM_STEPS_SCHNELL:-4}"
# 默认使用 GPU 6，可通过 BATCH_GPU 覆盖
BATCH_GPU="${BATCH_GPU:-6}"
CACHE_METHOD="${CACHE_METHOD:-none}" # none | hicache | eigencache
EIGENCACHE_KERNEL="${EIGENCACHE_KERNEL:-$PROJECT_ROOT/outputs/kernels.pt}" # used when cache_method=eigencache

export PROJECT_ROOT
export CUDA_VISIBLE_DEVICES="$BATCH_GPU"

cd "$PROJECT_ROOT"

for model in "${MODELS[@]}"; do
  if [[ "$model" == "flux-schnell" ]]; then
    steps=$NUM_STEPS_SCHNELL
  else
    steps=$NUM_STEPS_DEV
  fi

  for mode in "${MODES[@]}"; do
    out_dir="$OUTPUT_BASE/${model}/${mode}"
    mkdir -p "$out_dir"

    args=(
      bash scripts/sample.sh
      --mode "$mode"
      --model_name "$model"
      --prompt_file "$PROMPT_FILE"
      --output_dir "$out_dir"
      --interval "$INTERVAL"
      --max_order "$MAX_ORDER"
      --hicache_scale "$HICACHE_SCALE"
      --first_enhance "$FIRST_ENHANCE"
      --num_steps "$steps"
      --cache_method "$CACHE_METHOD"
    )

    if [[ "$CACHE_METHOD" == "eigencache" ]]; then
      args+=(
        --eigencache_kernel_path "$EIGENCACHE_KERNEL"
        --schedule fixed
        --eigencache_window_M 3
        --eigencache_lambda 1e-3
      )
    fi

    echo "[RUN] ${args[*]}"
    "${args[@]}"
  done
done
