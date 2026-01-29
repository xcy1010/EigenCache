#!/usr/bin/env bash

# Pre-processing: Set working directory and venv environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT" || { echo "[ERROR] Failed to enter project directory"; exit 1; }
echo "[INFO] Working directory: $(pwd)"

# Activate virtual environment
if [[ ! -d ".venv" ]]; then
    echo "[ERROR] .venv virtual environment does not exist, please create it first"
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate || { echo "[ERROR] Failed to activate .venv"; exit 1; }
echo "[INFO] Activated virtual environment: .venv"

# 0锔忊儯  Unify temp and cache directories to avoid root partition usage
DEFAULT_CACHE_ROOT="$PROJECT_ROOT/.cache"
export TEMP_ROOT="${TEMP_ROOT:-$DEFAULT_CACHE_ROOT}"
export TMPDIR="${TMPDIR:-$TEMP_ROOT}"
export TMP="${TMP:-$TEMP_ROOT}"
export TEMP="${TEMP:-$TEMP_ROOT}"
export HF_HOME="${HF_HOME:-$TEMP_ROOT/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$TEMP_ROOT/.huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$TEMP_ROOT/.transformers}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" || true

# 1锔忊儯  Ensure virtual environment is activated
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[ERROR] Virtual environment not activated, please check .venv"
    exit 1
fi

EXPECTED_VENV="$PROJECT_ROOT/.venv"
if [[ "$VIRTUAL_ENV" != "$EXPECTED_VENV" ]]; then
    echo "[ERROR] Virtual environment path mismatch"
    echo "       Current: $VIRTUAL_ENV"
    echo "       Expected: $EXPECTED_VENV"
    echo "       Recommend recreating virtual environment in new directory:"
    echo "         cd $PROJECT_ROOT && python3.10 -m venv .venv"
    echo "         source .venv/bin/activate && pip install -e \".[all]\""
    exit 1
fi

if [[ ! "$VIRTUAL_ENV" == *".venv"* ]]; then
    echo "[WARNING] Current environment ($VIRTUAL_ENV) may not be the expected .venv"
fi

echo "[INFO] Using virtual environment: $VIRTUAL_ENV"

# Default values
MODE="Taylor" # Taylor, Taylor-Scaled, HiCache, original, ToCa, Delta, collect, ClusCa, Hi-ClusCa
MODEL_NAME="flux-dev" # flux-dev | flux-schnell
INTERVAL="5"
MAX_ORDER="2"
OUTPUT_DIR="$PROJECT_ROOT/results"
PROMPT_FILE="./data/prompt.txt"
WIDTH=1024
HEIGHT=1024
NUM_STEPS=50
NUM_STEPS_SET=false
LIMIT=10
HICACHE_SCALE_FACTOR="0.7"
START_INDEX=0
MODEL_DIR=""
FIRST_ENHANCE=3
# EigenCache options
CACHE_METHOD="none" # none | hicache | eigencache
SCHEDULE="fixed" # fixed | greedy | variance
EIGENCACHE_KERNEL_PATH=""
EIGENCACHE_CALIBRATE=false
EIGENCACHE_CALIB_PROMPTS=""
EIGENCACHE_CALIB_RUNS=16
EIGENCACHE_WINDOW_M=3
EIGENCACHE_LAMBDA=1e-3
EIGENCACHE_BUDGET_B=8
EIGENCACHE_VAR_TAU=0.05
EIGENCACHE_LAYER_WEIGHTS=""
EIGENCACHE_PHASE_BOUNDARIES=""
EIGENCACHE_PHASE_NAMES=""
EIGENCACHE_PRECOMPUTE_WEIGHTS=false
EIGENCACHE_KL_RANK=0
# Kalman-HiCache and other adaptive modes have been removed

# --- ClusCa Default Parameters ----------------------------------------------------------
CLUSCA_FRESH_THRESHOLD=5        # ClusCa fresh 闃堝€?CLUSCA_CLUSTER_NUM=16           # 鑱氱被鏁伴噺
CLUSCA_CLUSTER_METHOD="kmeans"  # 鑱氱被鏂规硶 (kmeans/kmeans++/random)
CLUSCA_K=1                      # 姣忎釜鑱氱被閫夋嫨鐨?fresh token 鏁?CLUSCA_PROPAGATION_RATIO=0.005  # 鐗瑰緛浼犳挱姣斾緥
# ------------------------------------------------------------------------------

# Help information
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --mode MODE           Cache mode (Taylor, Taylor-Scaled, HiCache, original, ToCa, Delta, collect, ClusCa, Hi-ClusCa) [default: Taylor]"
    echo "  --model_name NAME         FLUX model (flux-dev|flux-schnell) [default: flux-dev]"
    echo "  --model_dir DIR          Specify local FLUX weight directory (containing flow and ae)"
    echo "  -i, --interval INTERVAL   Interval value [default: 1]"
    echo "  -o, --max_order ORDER     Maximum order [default: 1]"
    echo "  -d, --output_dir DIR      Output directory [default: $PROJECT_ROOT/results]"
    echo "  -p, --prompt_file FILE    Prompt file [default: ./data/prompt.txt]"
    echo "  -w, --width WIDTH         Image width [default: 1024]"
    echo "  -h, --height HEIGHT       Image height [default: 1024]"
    echo "  -s, --num_steps STEPS     Sampling steps [default: 50]"
    echo "  -l, --limit LIMIT         Test quantity limit [default: 10]"
    echo "  --hicache_scale FACTOR    HiCache polynomial scaling factor [default: 0.5]"
    echo "  --first_enhance N         Initial enhancement steps (first N steps force full) [default: 3]"
    echo "  --start_index N            Result file number offset [default: 0]"
    echo "  --cache_method METHOD     Cache method override (none|hicache|eigencache) [default: none]"
    echo "  --schedule NAME           EigenCache schedule (fixed|greedy|variance) [default: fixed]"
    echo "  --eigencache_kernel_path PATH  EigenCache kernel file path"
    echo "  --eigencache_calibrate    Run EigenCache calibration and exit"
    echo "  --eigencache_calib_prompts FILE  Prompt file for calibration"
    echo "  --eigencache_calib_runs N  Calibration runs [default: 16]"
    echo "  --eigencache_window_M N    Anchor window size [default: 3]"
    echo "  --eigencache_lambda VAL    Kriging jitter [default: 1e-3]"
    echo "  --eigencache_budget_B N    Greedy schedule budget [default: 8]"
    echo "  --eigencache_var_tau VAL   Variance schedule threshold [default: 0.05]"
    echo "  --eigencache_layer_weights STR  Layer weights string"
    echo "  --eigencache_phase_boundaries STR  Phase boundaries (comma separated)"
    echo "  --eigencache_phase_names STR  Phase names (comma separated)"
    echo "  --eigencache_precompute_weights  Precompute Kriging weight tables"
    echo "  --eigencache_kl_rank N     KL truncation rank [default: 0]"
    echo "  --clusca_fresh_threshold N  ClusCa: fresh threshold [default: 5]"
    echo "  --clusca_cluster_num N    ClusCa: number of clusters [default: 16]"
    echo "  --clusca_cluster_method M ClusCa: clustering method (kmeans/kmeans++/random) [default: kmeans]"
    echo "  --clusca_k N              ClusCa: number of fresh tokens per cluster [default: 1]"
    echo "  --clusca_propagation_ratio R  ClusCa: feature propagation ratio [default: 0.005]"
    echo "  --help                    Show help information"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -o|--max_order)
            MAX_ORDER="$2"
            shift 2
            ;;
        -d|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--prompt_file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        -w|--width)
            WIDTH="$2"
            shift 2
            ;;
        -h|--height)
            HEIGHT="$2"
            shift 2
            ;;
        -s|--num_steps)
            NUM_STEPS="$2"
            NUM_STEPS_SET=true
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        --hicache_scale)
            HICACHE_SCALE_FACTOR="$2"
            shift 2
            ;;
        --cache_method)
            CACHE_METHOD="$2"
            shift 2
            ;;
        --schedule)
            SCHEDULE="$2"
            shift 2
            ;;
        --eigencache_kernel_path)
            EIGENCACHE_KERNEL_PATH="$2"
            shift 2
            ;;
        --eigencache_calibrate)
            EIGENCACHE_CALIBRATE=true
            shift
            ;;
        --eigencache_calib_prompts)
            EIGENCACHE_CALIB_PROMPTS="$2"
            shift 2
            ;;
        --eigencache_calib_runs)
            EIGENCACHE_CALIB_RUNS="$2"
            shift 2
            ;;
        --eigencache_window_M)
            EIGENCACHE_WINDOW_M="$2"
            shift 2
            ;;
        --eigencache_lambda)
            EIGENCACHE_LAMBDA="$2"
            shift 2
            ;;
        --eigencache_budget_B)
            EIGENCACHE_BUDGET_B="$2"
            shift 2
            ;;
        --eigencache_var_tau)
            EIGENCACHE_VAR_TAU="$2"
            shift 2
            ;;
        --eigencache_layer_weights)
            EIGENCACHE_LAYER_WEIGHTS="$2"
            shift 2
            ;;
        --eigencache_phase_boundaries)
            EIGENCACHE_PHASE_BOUNDARIES="$2"
            shift 2
            ;;
        --eigencache_phase_names)
            EIGENCACHE_PHASE_NAMES="$2"
            shift 2
            ;;
        --eigencache_precompute_weights)
            EIGENCACHE_PRECOMPUTE_WEIGHTS=true
            shift
            ;;
        --eigencache_kl_rank)
            EIGENCACHE_KL_RANK="$2"
            shift 2
            ;;
        --first_enhance)
            FIRST_ENHANCE="$2"
            shift 2
            ;;
        --start_index)
            START_INDEX="$2"
            shift 2
            ;;
        --clusca_fresh_threshold)
            CLUSCA_FRESH_THRESHOLD="$2"
            shift 2
            ;;
        --clusca_cluster_num)
            CLUSCA_CLUSTER_NUM="$2"
            shift 2
            ;;
        --clusca_cluster_method)
            CLUSCA_CLUSTER_METHOD="$2"
            shift 2
            ;;
        --clusca_k)
            CLUSCA_K="$2"
            shift 2
            ;;
        --clusca_propagation_ratio)
            CLUSCA_PROPAGATION_RATIO="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "Taylor" && "$MODE" != "Taylor-Scaled" && "$MODE" != "HiCache" && "$MODE" != "original" &&
    "$MODE" != "ToCa" && "$MODE" != "Delta" && "$MODE" != "collect" && "$MODE" != "ClusCa" && "$MODE" != "Hi-ClusCa" ]]; then
    echo "Error: Unsupported mode '$MODE'"
    echo "Supported modes: Taylor, Taylor-Scaled, HiCache, original, ToCa, Delta, collect, ClusCa, Hi-ClusCa"
    exit 1
fi

if [[ "$CACHE_METHOD" != "none" && "$CACHE_METHOD" != "hicache" && "$CACHE_METHOD" != "eigencache" ]]; then
    echo "Error: Unsupported cache_method '$CACHE_METHOD'"
    echo "Supported cache_method: none, hicache, eigencache"
    exit 1
fi

if [[ "$CACHE_METHOD" == "eigencache" && "$EIGENCACHE_CALIBRATE" != true && -z "$EIGENCACHE_KERNEL_PATH" ]]; then
    echo "[ERROR] EigenCache inference requires --eigencache_kernel_path"
    exit 1
fi

# Set environment variables
# Model path prioritizes local cache, fallback to remote weight names when not found
export T5_DIR="$PROJECT_ROOT/models/t5-v1_1-xxl"

resolve_clip_local() {
    local base_candidates=(
        "$PROJECT_ROOT/models/clip-vit-large-patch14"
        "$PROJECT_ROOT/models/clip-vit-large-patch14/clip-vit-large-patch14"
        "$PROJECT_ROOT/models/openai/clip-vit-large-patch14"
    )

    for candidate in "${base_candidates[@]}"; do
        if [[ -d "$candidate" && -f "$candidate/config.json" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    local hub_root="${HF_HOME:-$TEMP_ROOT/.hf_home}/hub/models--openai--clip-vit-large-patch14/snapshots"
    if [[ -d "$hub_root" ]]; then
        local latest
        latest=$(ls -1dt "$hub_root"/* 2>/dev/null | head -n1 || true)
        if [[ -n "$latest" && -f "$latest/config.json" ]]; then
            echo "$latest"
            return 0
        fi
    fi

    local found
    found=$(find "$PROJECT_ROOT/models" -type f -name config.json -path "*clip-vit-large-patch14*" 2>/dev/null | head -n1 || true)
    if [[ -n "$found" ]]; then
        echo "$(dirname "$found")"
        return 0
    fi

    return 1
}

CLIP_LOCAL_DIR="$(resolve_clip_local || true)"
if [[ -n "$CLIP_LOCAL_DIR" ]]; then
    export CLIP_DIR="$CLIP_LOCAL_DIR"
    export HF_HUB_OFFLINE="1"
    export HF_DATASETS_OFFLINE="1"
    export TRANSFORMERS_OFFLINE="1"
    echo "[INFO] Using local CLIP model: $CLIP_DIR"
else
    export CLIP_DIR="openai/clip-vit-large-patch14"
    export HF_HUB_OFFLINE="0"
    export HF_DATASETS_OFFLINE="0"
    export TRANSFORMERS_OFFLINE="0"
    echo "[WARN] Local CLIP cache not found, will try to download openai/clip-vit-large-patch14 online"
fi

# Auto match model directory path based on model name
auto_detect_model_dir() {
    local model_name="$1"
    local candidates=()

    if [[ "$model_name" == "flux-schnell" ]]; then
        candidates=(
            "$PROJECT_ROOT/models/FLUX.1-schnell"
            "$PROJECT_ROOT/models/flux.schnell"
            "$PROJECT_ROOT/models/flux-schnell"
            "$PROJECT_ROOT/models/schnell"
        )
    else
        candidates=(
            "$PROJECT_ROOT/models/FLUX.1-dev"
            "$PROJECT_ROOT/models/flux.dev"
            "$PROJECT_ROOT/models/flux-dev"
            "$PROJECT_ROOT/models/dev"
        )
    fi

    for candidate in "${candidates[@]}"; do
        if [[ -d "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

# Set model directory
if [[ -n "$MODEL_DIR" ]]; then
    echo "[INFO] Specified --model_dir: $MODEL_DIR"
    AUTO_MODEL_DIR="$MODEL_DIR"
else
    AUTO_MODEL_DIR="$(auto_detect_model_dir "$MODEL_NAME")"
    if [[ -z "$AUTO_MODEL_DIR" ]]; then
        echo "[ERROR] No matching model directory found, please check models/ directory or use --model_dir to specify"
        echo "Supported directory formats:"
        if [[ "$MODEL_NAME" == "flux-schnell" ]]; then
            echo "  - models/FLUX.1-schnell"
            echo "  - models/flux.schnell"
            echo "  - models/flux-schnell"
            echo "  - models/schnell"
        else
            echo "  - models/FLUX.1-dev"
            echo "  - models/flux.dev"
            echo "  - models/flux-dev"
            echo "  - models/dev"
        fi
        exit 1
    else
        echo "[INFO] Auto-detected model directory: $AUTO_MODEL_DIR"
    fi
fi

# Set weight path based on model name
if [[ "$MODEL_NAME" == "flux-schnell" ]]; then
    if [[ -f "$AUTO_MODEL_DIR/flux1-schnell.safetensors" ]]; then
        export FLUX_SCHNELL="$AUTO_MODEL_DIR/flux1-schnell.safetensors"
        echo "[INFO] Using FLUX_SCHNELL: $FLUX_SCHNELL"
    else
        echo "[ERROR] flux1-schnell.safetensors not found in $AUTO_MODEL_DIR"
        exit 1
    fi
else
    if [[ -f "$AUTO_MODEL_DIR/flux1-dev.safetensors" ]]; then
        export FLUX_DEV="$AUTO_MODEL_DIR/flux1-dev.safetensors"
        echo "[INFO] Using FLUX_DEV: $FLUX_DEV"
    else
        echo "[ERROR] flux1-dev.safetensors not found in $AUTO_MODEL_DIR"
        exit 1
    fi
fi

if [[ -f "$AUTO_MODEL_DIR/ae.safetensors" ]]; then
    export AE="$AUTO_MODEL_DIR/ae.safetensors"
    echo "[INFO] Using AE: $AE"
else
    echo "[ERROR] ae.safetensors not found in $AUTO_MODEL_DIR"
    exit 1
fi

# Before determining directory, if steps not explicitly specified and is schnell, set default steps to 4
if [[ "$MODEL_NAME" == "flux-schnell" && "$NUM_STEPS_SET" != true ]]; then
    NUM_STEPS=4
fi

# Unified output directory: only keep first-level directory as mode, merge other key parameters as subdirectory names
MODE_LOWER="${MODE,,}"
METHOD_TAG=""
if [[ "$CACHE_METHOD" != "none" ]]; then
    METHOD_TAG="_cm_${CACHE_METHOD}"
fi
if [[ "$CACHE_METHOD" == "eigencache" ]]; then
    METHOD_TAG="${METHOD_TAG}_sch_${SCHEDULE}"
fi
PARAM_TAG="mn_${MODEL_NAME}_i_${INTERVAL}_o_${MAX_ORDER}_s_${NUM_STEPS}_hs_${HICACHE_SCALE_FACTOR}${METHOD_TAG}"
FULL_OUTPUT_DIR="$OUTPUT_DIR/${MODE_LOWER}/${PARAM_TAG}"
mkdir -p "$FULL_OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo "$FULL_OUTPUT_DIR" > "$OUTPUT_DIR/.full_output_dir"

# Create limited temporary prompt file
TEMP_PROMPT_FILE="$FULL_OUTPUT_DIR/temp_prompts.txt"
head -n "$LIMIT" "$PROMPT_FILE" > "$TEMP_PROMPT_FILE"

# Show configuration information
echo "================================="
echo "Image generation configuration:"
echo "Mode: $MODE"
echo "Interval: $INTERVAL"
echo "Max order: $MAX_ORDER"
echo "Output directory: $FULL_OUTPUT_DIR"
echo "Prompt file: $PROMPT_FILE"
echo "FLUX model: $MODEL_NAME"
if [[ -n "$MODEL_DIR" ]]; then
    echo "Model directory: $MODEL_DIR"
elif [[ -n "$AUTO_MODEL_DIR" ]]; then
    echo "Auto-detected model directory: $AUTO_MODEL_DIR"
fi
if [[ "$MODEL_NAME" == "flux-schnell" && "$NUM_STEPS_SET" != true ]]; then
    # If user didn't explicitly specify steps, schnell default steps is 4
    NUM_STEPS=4
fi

echo "Image size: ${WIDTH}x${HEIGHT}"
echo "Sampling steps: $NUM_STEPS"
echo "Test quantity limit: $LIMIT"
echo "HiCache scaling factor: $HICACHE_SCALE_FACTOR"
echo "First enhance: $FIRST_ENHANCE"
echo "Start index: $START_INDEX"
echo "Cache method: $CACHE_METHOD"
if [[ "$CACHE_METHOD" == "eigencache" ]]; then
    echo "EigenCache schedule: $SCHEDULE"
    if [[ -n "$EIGENCACHE_KERNEL_PATH" ]]; then
        echo "EigenCache kernel: $EIGENCACHE_KERNEL_PATH"
    fi
fi
if [[ "$MODE" == "ClusCa" ]]; then
    echo "ClusCa fresh threshold: $CLUSCA_FRESH_THRESHOLD"
    echo "ClusCa clusters: $CLUSCA_CLUSTER_NUM ($CLUSCA_CLUSTER_METHOD)"
    echo "ClusCa k: $CLUSCA_K, propagation ratio: $CLUSCA_PROPAGATION_RATIO"
elif [[ "$MODE" == "Hi-ClusCa" ]]; then
    echo "Hi-ClusCa fresh threshold: $CLUSCA_FRESH_THRESHOLD"
    echo "Hi-ClusCa clusters: $CLUSCA_CLUSTER_NUM ($CLUSCA_CLUSTER_METHOD)"
    echo "Hi-ClusCa k: $CLUSCA_K, propagation ratio: $CLUSCA_PROPAGATION_RATIO"
    echo "Hi-ClusCa HiCache scaling: $HICACHE_SCALE_FACTOR"
fi
echo "================================="

CLUSCA_ARGS=()
if [[ "$MODE" == "ClusCa" || "$MODE" == "Hi-ClusCa" ]]; then
    CLUSCA_ARGS+=(--clusca_fresh_threshold "$CLUSCA_FRESH_THRESHOLD")
    CLUSCA_ARGS+=(--clusca_cluster_num "$CLUSCA_CLUSTER_NUM")
    CLUSCA_ARGS+=(--clusca_cluster_method "$CLUSCA_CLUSTER_METHOD")
    CLUSCA_ARGS+=(--clusca_k "$CLUSCA_K")
    CLUSCA_ARGS+=(--clusca_propagation_ratio "$CLUSCA_PROPAGATION_RATIO")
fi

EIGENCACHE_ARGS=(--cache_method "$CACHE_METHOD" --schedule "$SCHEDULE")
if [[ -n "$EIGENCACHE_KERNEL_PATH" ]]; then
    EIGENCACHE_ARGS+=(--eigencache_kernel_path "$EIGENCACHE_KERNEL_PATH")
fi
if [[ "$EIGENCACHE_CALIBRATE" == true ]]; then
    EIGENCACHE_ARGS+=(--eigencache_calibrate)
fi
if [[ -n "$EIGENCACHE_CALIB_PROMPTS" ]]; then
    EIGENCACHE_ARGS+=(--eigencache_calib_prompts "$EIGENCACHE_CALIB_PROMPTS")
fi
EIGENCACHE_ARGS+=(--eigencache_calib_runs "$EIGENCACHE_CALIB_RUNS")
EIGENCACHE_ARGS+=(--eigencache_window_M "$EIGENCACHE_WINDOW_M")
EIGENCACHE_ARGS+=(--eigencache_lambda "$EIGENCACHE_LAMBDA")
EIGENCACHE_ARGS+=(--eigencache_budget_B "$EIGENCACHE_BUDGET_B")
EIGENCACHE_ARGS+=(--eigencache_var_tau "$EIGENCACHE_VAR_TAU")
if [[ -n "$EIGENCACHE_LAYER_WEIGHTS" ]]; then
    EIGENCACHE_ARGS+=(--eigencache_layer_weights "$EIGENCACHE_LAYER_WEIGHTS")
fi
if [[ -n "$EIGENCACHE_PHASE_BOUNDARIES" ]]; then
    EIGENCACHE_ARGS+=(--eigencache_phase_boundaries "$EIGENCACHE_PHASE_BOUNDARIES")
fi
if [[ -n "$EIGENCACHE_PHASE_NAMES" ]]; then
    EIGENCACHE_ARGS+=(--eigencache_phase_names "$EIGENCACHE_PHASE_NAMES")
fi
if [[ "$EIGENCACHE_PRECOMPUTE_WEIGHTS" == true ]]; then
    EIGENCACHE_ARGS+=(--eigencache_precompute_weights)
fi
if [[ "$EIGENCACHE_KL_RANK" -gt 0 ]]; then
    EIGENCACHE_ARGS+=(--eigencache_kl_rank "$EIGENCACHE_KL_RANK")
fi

# Execute sampling
echo "Starting image generation..."
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="0"
fi
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python src/sample.py \
  --prompt_file "$TEMP_PROMPT_FILE" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --model_name "$MODEL_NAME" \
  --add_sampling_metadata \
  --output_dir "$FULL_OUTPUT_DIR" \
  --num_steps "$NUM_STEPS" \
  --cache_mode "$MODE" \
  --interval "$INTERVAL" \
  --max_order "$MAX_ORDER" \
  --first_enhance "$FIRST_ENHANCE" \
  --seed 0 \
  --start_index "$START_INDEX" \
  --hicache_scale "$HICACHE_SCALE_FACTOR" \
  "${CLUSCA_ARGS[@]}" \
  "${EIGENCACHE_ARGS[@]}"

PYTHON_EXIT_CODE=$?
if [[ $PYTHON_EXIT_CODE -ne 0 ]]; then
    echo "[ERROR] Image generation script execution failed (exit code: $PYTHON_EXIT_CODE)"
    rm -f "$TEMP_PROMPT_FILE"
    exit $PYTHON_EXIT_CODE
fi

echo "Image generation completed!"
echo "Output directory: $FULL_OUTPUT_DIR"

# Clean up temporary files
rm -f "$TEMP_PROMPT_FILE"

echo ""
# If in multi-GPU temp directory, avoid printing misleading evaluation commands (aggregator will give final recommendations)
case "$OUTPUT_DIR" in
  *"/.multi_gpu_tmp/"*)
    :
    ;;
  *)
    echo "================================="
    # Fixed GT recommended directory as Taylor baseline interval_1/order_2 (single-GPU scenario)
    GT_SUGGEST="$PROJECT_ROOT/results/taylor/interval_1/order_2"
    echo "Recommended evaluation command:"
    echo "  bash \"$PROJECT_ROOT/evaluation/run_eval.sh\" --acc \"$FULL_OUTPUT_DIR\" --gt \"$GT_SUGGEST\""
    echo "================================="
    ;;
esac

