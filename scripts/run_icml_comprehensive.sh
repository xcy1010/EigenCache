#!/bin/bash
set -e  # 遇到错误立即停止

# ================= 配置区域 =================
export PROJECT_ROOT=$(pwd)
export CUDA_VISIBLE_DEVICES=6  # 请根据实际情况修改显卡编号

# 输入与输出
PROMPT_FILE="data/prompt_icml_comprehensive_163.txt"
BASE_OUT_DIR="results_comprehensive_163"
LOG_DIR="logs_comprehensive_163"

# 统一超参 (FLUX-dev) - 激进参数，旨在拉大方法间差距
MODEL="flux-dev"
STEPS=50
WIDTH=1024
HEIGHT=1024
INTERVAL=10        # 激进：增大到 10，强迫模型更依赖缓存
MAX_ORDER=2
FIRST_ENHANCE=2    # 激进：减小到 2，让误差更早介入
HICACHE_SCALE=0.6  # 保持 0.6 以维持锐度

# 创建目录
mkdir -p "$BASE_OUT_DIR" "$LOG_DIR" "outputs"

echo "========================================================"
echo "🚀 开始 ICML Comprehensive 实验 (163 Prompts, 激进参数)"
echo "显卡: $CUDA_VISIBLE_DEVICES | 输出目录: $BASE_OUT_DIR"
echo "参数: Interval=$INTERVAL | Enhance=$FIRST_ENHANCE | Scale=$HICACHE_SCALE"
echo "========================================================"

# ------------------------------------------------------
# 1. GT (Original) - 全量推理基准
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
# 7. EigenCache - Calibration (校准)
# ------------------------------------------------------
KERNEL_PATH="outputs/kernel_comprehensive_i${INTERVAL}.pt"
echo "[7/8] Running EigenCache Calibration (Generating Kernel)..."
# 注意：这里直接使用 src/sample.py 以便传入特定参数
# 校准时使用前 30 条 prompt 即可，节省时间
CUDA_VISIBLE_DEVICES=6 python src/sample.py \
    --prompt_file "$PROMPT_FILE" --limit 30 --model_name $MODEL \
    --cache_mode Taylor --cache_method eigencache --schedule fixed \
    --interval $INTERVAL --max_order $MAX_ORDER --first_enhance $FIRST_ENHANCE --hicache_scale $HICACHE_SCALE \
    --num_steps $STEPS --width $WIDTH --height $HEIGHT \
    --eigencache_calibrate --eigencache_calib_prompts "$PROMPT_FILE" --eigencache_calib_runs 30 \
    --eigencache_kernel_path "$KERNEL_PATH" 2>&1 | tee "$LOG_DIR/eigencache_calib.log"

# ------------------------------------------------------
# 8. EigenCache - Inference (推理)
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
echo "✅ 所有实验运行完毕！"
echo "结果目录: $BASE_OUT_DIR"
echo "日志目录: $LOG_DIR"
echo "========================================================"