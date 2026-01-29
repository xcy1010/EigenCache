#!/bin/bash
set -e  # Stop on error

# ================= Configuration Area =================
export PROJECT_ROOT=$(pwd)
export CUDA_VISIBLE_DEVICES=6  # Please modify the GPU ID according to your actual situation

# Input and Output
PROMPT_FILE="data/prompt_icml_comprehensive_163.txt"
BASE_OUT_DIR="results_comprehensive_163"
LOG_DIR="logs_comprehensive_163"

# Unified Hyperparameters (FLUX-dev) - Aggressive parameters to maximize differences between methods
MODEL="flux-dev"
STEPS=50
WIDTH=1024
HEIGHT=1024
INTERVAL=10        # Aggressive: Increase to 10 to force the model to rely more on cache
MAX_ORDER=2
FIRST_ENHANCE=2    # Aggressive: Decrease to 2 to allow errors to intervene earlier
HICACHE_SCALE=0.6  # Keep at 0.6 to maintain sharpness

# Create directories
mkdir -p "$BASE_OUT_DIR" "$LOG_DIR" "outputs"

echo "========================================================"
echo "🚀 Starting ICML Comprehensive Experiment (163 Prompts, Aggressive Parameters)"
echo "GPU: $CUDA_VISIBLE_DEVICES | Output Directory: $BASE_OUT_DIR"
echo "Parameters: Interval=$INTERVAL | Enhance=$FIRST_ENHANCE | Scale=$HICACHE_SCALE"
echo "========================================================"

# ------------------------------------------------------
# 1. GT (Original) - Full inference baseline
# ------------------------------------------------------
echo "[1/8] Running Original (GT)..."
CUDA_VISIBLE_DEVICES=6 /usr/bin/time -v bash scripts/sample.sh \
    --mode original --model_name $MODEL \
    --prompt_file "$PROMPT_FILE" --limit 163 \
    --width $WIDTH --height $HEIGHT --num_steps $STEPS \
    --interval $INTERVAL --max_order $MAX_ORDER --first_enhance $FIRST_ENHANCE --hicache_scale $HICACHE_SCALE \
    --output_dir "$BASE_OUT_DIR" 2>&1 | tee "$LOG_DIR/original.log"

# ------------------------------------------------------
# 2. ToCa
# ------------------------------------------------------
echo "[2/8] Running ToCa..."
CUDA_VISIBLE_DEVICES=6 /usr/bin/time -v bash scripts/sample.sh \
    --mode ToCa --model_name $MODEL \
    --prompt_file "$PROMPT_FILE" --limit 163 \
    --width $WIDTH --height $HEIGHT --num_steps $STEPS \
    --interval $INTERVAL --max_order $MAX_ORDER --first_enhance $FIRST_ENHANCE --hicache_scale $HICACHE_SCALE \
    --output_dir "$BASE_OUT_DIR" 2>&1 | tee "$LOG_DIR/toca.log"

# ------------------------------------------------------
# 3. Delta
# ------------------------------------------------------
echo "[3/8] Running Delta..."
CUDA_VISIBLE_DEVICES=6 /usr/bin/time -v bash scripts/sample.sh \
    --mode Delta --model_name $MODEL \
    --prompt_file "$PROMPT_FILE" --limit 163 \
    --width $WIDTH --height $HEIGHT --num_steps $STEPS \
    --interval $INTERVAL --max_order $MAX_ORDER --first_enhance $FIRST_ENHANCE --hicache_scale $HICACHE_SCALE \
    --output_dir "$BASE_OUT_DIR" 2>&1 | tee "$LOG_DIR/delta.log"

# ------------------------------------------------------
# 4. Taylor
# ------------------------------------------------------
echo "[4/8] Running Taylor..."
CUDA_VISIBLE_DEVICES=6 /usr/bin/time -v bash scripts/sample.sh \
    --mode Taylor --model_name $MODEL \
    --prompt_file "$PROMPT_FILE" --limit 163 \
    --width $WIDTH --height $HEIGHT --num_steps $STEPS \
    --interval $INTERVAL --max_order $MAX_ORDER --first_enhance $FIRST_ENHANCE --hicache_scale $HICACHE_SCALE \
    --output_dir "$BASE_OUT_DIR" 2>&1 | tee "$LOG_DIR/taylor.log"

# ------------------------------------------------------
# 5. HiCache (Ours)
# ------------------------------------------------------
echo "[5/8] Running HiCache..."
CUDA_VISIBLE_DEVICES=6 /usr/bin/time -v bash scripts/sample.sh \
    --mode HiCache --model_name $MODEL \
    --prompt_file "$PROMPT_FILE" --limit 163 \
    --width $WIDTH --height $HEIGHT --num_steps $STEPS \
    --interval $INTERVAL --max_order $MAX_ORDER --first_enhance $FIRST_ENHANCE --hicache_scale $HICACHE_SCALE \
    --output_dir "$BASE_OUT_DIR" 2>&1 | tee "$LOG_DIR/hicache.log"

# ------------------------------------------------------
# 6. ClusCa
# ------------------------------------------------------
echo "[6/8] Running ClusCa..."
CUDA_VISIBLE_DEVICES=6 /usr/bin/time -v bash scripts/sample.sh \
    --mode ClusCa --model_name $MODEL \
    --prompt_file "$PROMPT_FILE" --limit 163 \
    --width $WIDTH --height $HEIGHT --num_steps $STEPS \
    --interval $INTERVAL --max_order $MAX_ORDER --first_enhance $FIRST_ENHANCE --hicache_scale $HICACHE_SCALE \
    --clusca_cluster_num 16 --clusca_k 1 --clusca_propagation_ratio 0.005 \
    --output_dir "$BASE_OUT_DIR" 2>&1 | tee "$LOG_DIR/clusca.log"

# ------------------------------------------------------
# 7. EigenCache - Calibration
# ------------------------------------------------------
KERNEL_PATH="outputs/kernel_comprehensive_i${INTERVAL}.pt"
echo "[7/8] Running EigenCache Calibration (Generating Kernel)..."
# Note: Using src/sample.py directly to pass specific parameters
# Use the first 30 prompts for calibration to save time, which is sufficient to cover the distribution
CUDA_VISIBLE_DEVICES=6 python src/sample.py \
    --prompt_file "$PROMPT_FILE" --limit 30 --model_name $MODEL \
    --cache_mode Taylor --cache_method eigencache --schedule fixed \
    --interval $INTERVAL --max_order $MAX_ORDER --first_enhance $FIRST_ENHANCE --hicache_scale $HICACHE_SCALE \
    --num_steps $STEPS --width $WIDTH --height $HEIGHT \
    --eigencache_calibrate --eigencache_calib_prompts "$PROMPT_FILE" --eigencache_calib_runs 30 \
    --eigencache_kernel_path "$KERNEL_PATH" 2>&1 | tee "$LOG_DIR/eigencache_calib.log"

# ------------------------------------------------------
# 8. EigenCache - Inference
# ------------------------------------------------------
echo "[8/8] Running EigenCache Inference (Fixed Schedule)..."
CUDA_VISIBLE_DEVICES=6 /usr/bin/time -v bash scripts/sample.sh \
    --mode Taylor --model_name $MODEL \
    --prompt_file "$PROMPT_FILE" --limit 163 \
    --width $WIDTH --height $HEIGHT --num_steps $STEPS \
    --interval $INTERVAL --max_order $MAX_ORDER --first_enhance $FIRST_ENHANCE --hicache_scale $HICACHE_SCALE \
    --cache_method eigencache --schedule fixed \
    --eigencache_kernel_path "$KERNEL_PATH" --eigencache_window_M 3 --eigencache_lambda 1e-3 \
    --output_dir "$BASE_OUT_DIR" 2>&1 | tee "$LOG_DIR/eigencache_fixed.log"

echo "========================================================"
echo "✅ All experiments completed!"
echo "Results directory: $BASE_OUT_DIR"
echo "Log directory: $LOG_DIR"
echo "========================================================"