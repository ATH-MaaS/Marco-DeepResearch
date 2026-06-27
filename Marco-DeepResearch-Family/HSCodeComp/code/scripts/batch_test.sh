#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# HSCode Frontier Test — 批量测试启动脚本
# ═══════════════════════════════════════════════════════════════
#
# Usage:
#   ./scripts/batch_test.sh                    # 默认 qwen3.7-max，全量 633 题
#   ./scripts/batch_test.sh --model glm-5.2    # 切换到 GLM-5.2
#   ./scripts/batch_test.sh --limit 10         # 冒烟测试 10 题
#   ./scripts/batch_test.sh --parallel 8       # 8 并发
#
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  🏷️  HSCode Frontier Test — Batch Classification"
echo "  📁 Project: $PROJECT_DIR"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Default args
MODEL="${MODEL:-}"
LIMIT="${LIMIT:-}"
PARALLEL="${PARALLEL:-4}"

# Parse convenience env vars into CLI args
EXTRA_ARGS=()
if [ -n "$MODEL" ]; then
    EXTRA_ARGS+=(--model "$MODEL")
fi
if [ -n "$LIMIT" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi
EXTRA_ARGS+=(--parallel "$PARALLEL")

# Pass through all script arguments
python batch_test.py "${EXTRA_ARGS[@]}" "$@"
