#!/bin/bash
# Search2Skill local training launcher (single node, 8 GPUs)
# Usage:
#   cd Search2Skill/verl
#   bash scripts/train_xy/run_local.sh
#
# Prerequisites:
#   1. Install verl:  pip install -e .
#   2. Install deps:  pip install -r scripts/train_xy/requirements_appbuilder_search_visit.txt
#   3. Install flash_attn: pip install flash_attn
#   4. Set up Search2Skill config (../config/config.yaml) with your LLM API endpoints
#   5. Start Ray:  ray start --head
#
# Key environment variables (override as needed):
#   MODEL_PATH       - Path to the SFT base model (required)
#   DATA_PATH        - Path to the training data directory (default: data/exskill_similar2)
#   SAVE_PATH        - Checkpoint save directory (default: ./checkpoints/<experiment_name>)
#   NUM_GPU_PER_NODE - Number of GPUs (default: 8)
#   SEARCH2SKILL_CONFIG_PATH - Path to Search2Skill config YAML

set -x

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

# ===================== Configuration =====================
export MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set to the SFT base model directory}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/exskill_similar2}"
export SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/checkpoints/search2skill_training}"

export PROJECT_NAME="${PROJECT_NAME:-search2skill_grpo}"
export JOB_NAME="${JOB_NAME:-search2skill_8b_training}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-2}"
export NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-8}"
export WORLD_SIZE="${WORLD_SIZE:-1}"

export ROLLOUT_DP="${ROLLOUT_DP:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export GPU_UTIL="${GPU_UTIL:-0.85}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$((18 * 1024 - 1))}"

export TEST_FREQ="${TEST_FREQ:-5}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-30}"

# Search2Skill config
export SEARCH2SKILL_CONFIG_PATH="${SEARCH2SKILL_CONFIG_PATH:-${REPO_ROOT}/../config/config.yaml}"
export EXSKILL_CONFIG_PATH="${SEARCH2SKILL_CONFIG_PATH}"
export PYTHONPATH="${REPO_ROOT}/..:${PYTHONPATH}"

export EXSKILL_CACHE_DIR="${EXSKILL_CACHE_DIR:-${REPO_ROOT}/cache}"
export EXSKILL_RAW_PAGE_CACHE_ENABLED="${EXSKILL_RAW_PAGE_CACHE_ENABLED:-true}"
export EXSKILL_RAW_PAGE_CACHE_DIR="${EXSKILL_CACHE_DIR}/raw_page_cache_v2"
export EXSKILL_GOOGLE_SEARCH_CACHE_DIR="${EXSKILL_CACHE_DIR}/google_search_cache_v2"

# Skill quality / reuse evaluation
export SKILL_QUALITY_ENABLED="${SKILL_QUALITY_ENABLED:-true}"
export SKILL_REUSE_LLM_PROFILE="${SKILL_REUSE_LLM_PROFILE:-frozen_qwen3_8b_local}"
export SKILL_REUSE_BASELINE_PATH="${SKILL_REUSE_BASELINE_PATH:-${DATA_PATH}/baseline_pass_rates.json}"

# W&B (set your own key, or disable with trainer.logger='["console"]')
export WANDB_API_KEY="${WANDB_API_KEY:-<YOUR_WANDB_API_KEY>}"

# ===================== Compute steps =====================
get_parquet_count() {
    $PYTHON -c "import pandas as pd; print(len(pd.read_parquet('$1')))"
}

STEPS_PER_EPOCH=$(get_parquet_count "${DATA_PATH}/train.parquet")
STEPS_PER_EPOCH=$((STEPS_PER_EPOCH / TRAIN_BATCH_SIZE))
export STEPS_PER_EPOCH

SAVE_FREQ=$((STEPS_PER_EPOCH / 6))
if [ "$SAVE_FREQ" -lt 1 ]; then SAVE_FREQ=1; fi
export SAVE_FREQ

echo "========================================="
echo "Search2Skill GRPO Training"
echo "  MODEL_PATH:    ${MODEL_PATH}"
echo "  DATA_PATH:     ${DATA_PATH}"
echo "  SAVE_PATH:     ${SAVE_PATH}"
echo "  GPUs:          ${NUM_GPU_PER_NODE}"
echo "  STEPS/EPOCH:   ${STEPS_PER_EPOCH}"
echo "  TOTAL_EPOCHS:  ${TOTAL_EPOCHS}"
echo "========================================="

mkdir -p "$SAVE_PATH" "$EXSKILL_CACHE_DIR"

# ===================== Launch training =====================
bash scripts/train_xy/run_qwen3-8b_exskill_multiturn.sh
