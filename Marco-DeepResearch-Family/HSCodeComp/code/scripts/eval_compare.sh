#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# 评估脚本 — 对比两个模型的结果
# ═══════════════════════════════════════════════════════════════
#
# Usage:
#   ./scripts/eval_compare.sh output/hscode_qwen*.jsonl output/hscode_glm*.jsonl
#
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

python eval.py --compare "$@"
