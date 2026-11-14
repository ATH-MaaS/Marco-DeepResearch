#!/bin/bash
# Unified evaluation runner for Search2Skill
# Runs eval_benchmark.py across multiple datasets and modes.
#
# Usage:
#   bash scripts/run_all_eval.sh [MODEL] [CONFIG]
#   bash scripts/run_all_eval.sh qwen3-8b-local config/config.yaml
#
# Env vars:
#   MAIN_MODES    override eval mode (default: "search_skill free_tool direct")

set -e

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

MODEL="${1:-qwen3-8b-local}"
CONFIG="${2:-config/config.yaml}"

# Environment — adjust to your own Python path
PYTHON="${PYTHON:-python}"
cd "$(dirname "$0")/.."

# --- Main agent (eval_benchmark.py) ---
MAIN_SCRIPT="scripts/eval_benchmark.py"
MAIN_MODES="${MAIN_MODES:-search_skill free_tool direct}"
SKILL_LIBRARY="${SKILL_LIBRARY:-}"
CONCURRENCY=14
MAX_ROUNDS=15

# Data files and corresponding benchmark names
DATA_FILES=(
    "data/mmlu_pro_philosophy/philosophy_all.json"
    "data/mmlu_pro_law/law_all.json"
    "data/mmlu_pro_history/history_all.json"
    "data/supergpqa_baseline/supergpqa_hard_mathematics_500.json"
    "data/supergpqa_baseline/supergpqa_mixed_management_500.json"
    "data/supergpqa_baseline/supergpqa_middle_science_500.json"
)
BENCHMARKS=(
    "mmlu_pro_philosophy"
    "mmlu_pro_law"
    "mmlu_pro_history"
    "supergpqa"
    "supergpqa"
    "supergpqa"
)

# --- Execution settings ---
PARALLEL=${PARALLEL:-3}

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

OUTPUT_BASE="${RUN_OUTPUT_BASE:-results/direct_eval}"
mkdir -p "${OUTPUT_BASE}"

LOG_DIR="${RUN_LOG_DIR:-${OUTPUT_BASE}}"
mkdir -p "${LOG_DIR}"

ORDER_TAGS_CSV="${EVAL_ORDER_TAGS:-original}"
ORDER_SEEDS_CSV="${EVAL_ORDER_SEEDS:-none}"
IFS=',' read -ra ORDER_TAGS <<< "${ORDER_TAGS_CSV}"
IFS=',' read -ra ORDER_SEEDS <<< "${ORDER_SEEDS_CSV}"
if [ "${#ORDER_TAGS[@]}" -ne "${#ORDER_SEEDS[@]}" ]; then
    echo "[ERROR] EVAL_ORDER_TAGS and EVAL_ORDER_SEEDS must have the same number of entries."
    exit 1
fi

echo "=============================================="
echo "  Search2Skill Evaluation Runner"
echo "  Model: ${MODEL}"
echo "  Config: ${CONFIG}"
echo "  Modes: ${MAIN_MODES}"
echo "  Data files: ${#DATA_FILES[@]}"
echo "  Order runs: ${ORDER_TAGS[*]}"
echo "  Parallel: ${PARALLEL}"
echo "  Log dir: ${LOG_DIR}"
echo "  Start time: $(date)"
echo "=============================================="

for order_idx in "${!ORDER_TAGS[@]}"; do
    ORDER_TAG="${ORDER_TAGS[$order_idx]}"
    SHUFFLE_SEED="${ORDER_SEEDS[$order_idx]}"
    ORDER_OUTPUT_BASE="${OUTPUT_BASE}/${ORDER_TAG}"
    ORDER_LOG_DIR="${LOG_DIR}/${ORDER_TAG}"
    OUTPUT_ARGS=(--output "${ORDER_OUTPUT_BASE}")
    SHUFFLE_ARGS=()
    if [[ -n "${SHUFFLE_SEED}" && "${SHUFFLE_SEED}" != "none" && "${SHUFFLE_SEED}" != "original" ]]; then
        SHUFFLE_ARGS=(--shuffle_seed "${SHUFFLE_SEED}")
    fi
    mkdir -p "${ORDER_OUTPUT_BASE}" "${ORDER_LOG_DIR}"

    echo ""
    echo "=============================================="
    echo "  Order run: ${ORDER_TAG}"
    echo "  Output base: ${ORDER_OUTPUT_BASE}"
    echo "=============================================="

    TOTAL=${#DATA_FILES[@]}
    for ((batch_start=0; batch_start<TOTAL; batch_start+=PARALLEL)); do
        PIDS=()
        NAMES=()
        LOG_FILES=()

        for ((j=batch_start; j<batch_start+PARALLEL && j<TOTAL; j++)); do
            DATA_FILE="${DATA_FILES[$j]}"
            BENCHMARK="${BENCHMARKS[$j]}"

            DATA_FILE_ARGS=()
            if [[ -n "${DATA_FILE}" ]]; then
                BASENAME=$(basename "${DATA_FILE}" .jsonl)
                BASENAME=$(basename "${BASENAME}" .json)
                DATA_FILE_ARGS=(--data_file "${DATA_FILE}")
            else
                BASENAME="${BENCHMARK}"
            fi
            LOG_FILE="${ORDER_LOG_DIR}/${BENCHMARK}/${BASENAME}_${MODEL}_${MAIN_MODES// /_}.log"
            mkdir -p "${ORDER_LOG_DIR}/${BENCHMARK}"

            echo "  [$((j+1))/${TOTAL}] ${BASENAME} (${BENCHMARK})"

            SKILL_LIB_ARGS=()
            if [[ -n "${SKILL_LIBRARY}" ]]; then
                SKILL_LIB_ARGS=(--skill_library "${SKILL_LIBRARY}")
            fi

            $PYTHON $MAIN_SCRIPT \
                --benchmark "${BENCHMARK}" \
                --mode ${MAIN_MODES} \
                --model "${MODEL}" \
                "${DATA_FILE_ARGS[@]}" \
                "${SKILL_LIB_ARGS[@]}" \
                "${SHUFFLE_ARGS[@]}" \
                --config "${CONFIG}" \
                --concurrency ${CONCURRENCY} \
                --max_rounds ${MAX_ROUNDS} \
                --num -1 \
                "${OUTPUT_ARGS[@]}" \
                > "${LOG_FILE}" 2>&1 &

            PIDS+=($!)
            NAMES+=("${BASENAME}")
            LOG_FILES+=("${LOG_FILE}")
        done

        # Wait for batch
        for k in "${!PIDS[@]}"; do
            pid="${PIDS[$k]}"
            name="${NAMES[$k]}"
            log_file="${LOG_FILES[$k]}"
            if wait "$pid"; then
                echo "  [DONE] ${name} (pid=${pid})"
            else
                echo "  [FAIL] ${name} (pid=${pid}) — see ${log_file}"
            fi
        done
        echo "  [BATCH] Completed at $(date)"
    done
done

echo ""
echo "=============================================="
echo "  All evaluations completed!"
echo "  Logs: ${LOG_DIR}"
echo "  End time: $(date)"
echo "=============================================="
