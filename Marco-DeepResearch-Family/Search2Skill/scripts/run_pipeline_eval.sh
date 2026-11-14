#!/bin/bash
# Pipeline evaluation runner: two-phase (train skills -> test) benchmarks
#
# Usage:
#   bash scripts/run_pipeline_eval.sh [MODEL] [CONFIG]
#   bash scripts/run_pipeline_eval.sh qwen3-8b-local config/config.yaml
#
# Each pipeline runs sequentially (uses internal concurrency).

set -e

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

MODEL="${1:-qwen3-8b-local}"
CONFIG="${2:-config/config.yaml}"

# Environment — adjust to your own Python path
PYTHON="${PYTHON:-python}"
cd "$(dirname "$0")/.."

PIPELINE_SCRIPT="scripts/eval_pipeline.py"
CONCURRENCY=8
DATASET_BATCH_SIZE=4
MAX_ROUNDS=20
TRAIN_MODE="search_skill"
TEST_MODES="direct skill_only"

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE DEFINITIONS
# Each entry: BENCHMARK|TRAIN_FILE|TEST_FILE|SPLIT_FILE|DATA_PATH
# Strategy A (explicit files): set TRAIN_FILE + TEST_FILE
# Strategy B (split file): set SPLIT_FILE + DATA_PATH
# ═══════════════════════════════════════════════════════════════════════════

PIPELINES=(
  "evoagentbench|data/evoagentbench/train.jsonl|data/evoagentbench/test.jsonl||"
  "livecode_bench|||data/livecode/task_split.json|data/livecode/"
  "mmlu_pro_history|data/mmlu_pro_history/history_train.json|data/mmlu_pro_history/history_test.json||"
  "mmlu_pro_law|data/mmlu_pro_law/law_train.json|data/mmlu_pro_law/law_test.json||"
  "mmlu_pro_philosophy|data/mmlu_pro_philosophy/philosophy_train.json|data/mmlu_pro_philosophy/philosophy_test.json||"
  "supergpqa_hard_mathematics|data/supergpqa_reuse_dataset/hard_mathematics/hard_mathematics_train.json|data/supergpqa_reuse_dataset/hard_mathematics/hard_mathematics_test.json||"
  "science_p_c|data/supergpqa_reuse_dataset/science_p_c/science_p_c_train.json|data/supergpqa_reuse_dataset/science_p_c/science_p_c_test.json||"
)

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT_BASE="results/pipeline_eval"

echo "=============================================="
echo "  Pipeline Evaluation Runner"
echo "  Model: ${MODEL}"
echo "  Train mode: ${TRAIN_MODE}"
echo "  Test modes: ${TEST_MODES}"
echo "  Pipelines: ${#PIPELINES[@]}"
echo "  Output base: ${OUTPUT_BASE}"
echo "  Start time: $(date)"
echo "=============================================="

TOTAL=${#PIPELINES[@]}
for ((batch_start=0; batch_start<TOTAL; batch_start+=DATASET_BATCH_SIZE)); do
    PIDS=()
    LABELS=()
    LOGS=()

    batch_end=$((batch_start + DATASET_BATCH_SIZE))
    if [[ ${batch_end} -gt ${TOTAL} ]]; then
        batch_end=${TOTAL}
    fi

    for ((i=batch_start; i<batch_end; i++)); do
        IFS='|' read -r BENCHMARK TRAIN_FILE TEST_FILE SPLIT_FILE DATA_PATH <<< "${PIPELINES[$i]}"

        OUTPUT_DIR="${OUTPUT_BASE}/${BENCHMARK}/${MODEL}"
        LOG_FILE="${OUTPUT_DIR}/run.log"
        if [[ -f "${LOG_FILE}" ]] && grep -q "Pipeline complete" "${LOG_FILE}"; then
            echo "  [SKIP] ${BENCHMARK} already completed (log: ${LOG_FILE})"
            continue
        fi
        mkdir -p "${OUTPUT_DIR}"

        echo ""
        echo "======================================================================"
        echo "  [LAUNCH $((i+1))/${TOTAL}] ${BENCHMARK}"
        echo "  Train: ${TRAIN_FILE:-'(via split file)'}"
        echo "  Test:  ${TEST_FILE:-'(via split file)'}"
        echo "  Time:  $(date)"
        echo "======================================================================"

        CMD="$PYTHON $PIPELINE_SCRIPT --benchmark ${BENCHMARK} --model ${MODEL} --config ${CONFIG}"
        CMD="${CMD} --train_mode ${TRAIN_MODE} --test_mode ${TEST_MODES}"
        CMD="${CMD} --concurrency ${CONCURRENCY} --max_rounds ${MAX_ROUNDS}"

        if [[ -n "${TRAIN_FILE}" && -n "${TEST_FILE}" ]]; then
            CMD="${CMD} --train_file \"${TRAIN_FILE}\" --test_file \"${TEST_FILE}\""
        elif [[ -n "${SPLIT_FILE}" && -n "${DATA_PATH}" ]]; then
            CMD="${CMD} --split_file \"${SPLIT_FILE}\" --data_path \"${DATA_PATH}\""
        else
            echo "  [ERROR] Pipeline ${BENCHMARK}: must specify train_file+test_file or split_file+data_path"
            continue
        fi

        echo "  CMD: ${CMD}"
        echo "  Log: ${LOG_FILE}"

        eval ${CMD} > "${LOG_FILE}" 2>&1 &
        PIDS+=($!)
        LABELS+=("${BENCHMARK}")
        LOGS+=("${LOG_FILE}")
    done

    for j in "${!PIDS[@]}"; do
        if wait "${PIDS[$j]}"; then
            echo "  [DONE] ${LABELS[$j]} completed successfully (log: ${LOGS[$j]})"
        else
            EC=$?
            echo "  [FAIL] ${LABELS[$j]} exited with code ${EC} — see ${LOGS[$j]}"
        fi
    done
done

echo ""
echo "=============================================="
echo "  All pipelines completed!"
echo "  Output base: ${OUTPUT_BASE}"
echo "  End time: $(date)"
echo "=============================================="
