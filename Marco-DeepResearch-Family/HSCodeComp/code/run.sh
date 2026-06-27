#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# HSCodeComp Agent — 一键推理 + 评估脚本
# ═══════════════════════════════════════════════════════════════
# Usage:
#   ./run.sh                          # 默认: qwen3.7-max, 并发4, 全量
#   ./run.sh -m glm-5.2 -p 8         # GLM-5.2, 并发8
#   ./run.sh -m claude-opus-4-8 -p 4  # Claude Opus 4.8 (Anthropic API)
#   ./run.sh -m gpt-5.5 -p 4         # GPT-5.5 (OpenAI API)
#   ./run.sh -n 10                    # 冒烟测试(10题)
#   ./run.sh --resume                 # 续跑(跳过已完成的)
#   ./run.sh --eval-only -o output/xxx.jsonl  # 仅评估已有结果

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 默认参数 ──
MODEL="${MODEL:-qwen3.7-max}"
PARALLEL="${PARALLEL:-4}"
LIMIT=""
RESUME=""
EVAL_ONLY=""
OUTPUT=""
EXTRA_ARGS=""

# ── 解析参数 ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)     MODEL="$2"; shift 2 ;;
        -p|--parallel)  PARALLEL="$2"; shift 2 ;;
        -n|--limit)     LIMIT="$2"; shift 2 ;;
        -o|--output)    OUTPUT="$2"; shift 2 ;;
        --resume)       RESUME="--resume"; shift ;;
        --eval-only)    EVAL_ONLY="--eval-only"; shift ;;
        --verbose|-v)   EXTRA_ARGS="$EXTRA_ARGS -v"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -m, --model MODEL      模型名称 (default: qwen3.7-max)"
            echo "  -p, --parallel N       并发数 (default: 4)"
            echo "  -n, --limit N          限制题数 (冒烟测试)"
            echo "  -o, --output PATH      输出路径 (default: output/<model>_<timestamp>.jsonl)"
            echo "  --resume               续跑模式 (跳过已完成 case)"
            echo "  --eval-only            仅评估已有结果"
            echo "  -v, --verbose          详细日志"
            echo "  -h, --help             显示帮助"
            echo ""
            echo "Environment Variables:"
            echo "  DASHSCOPE_API_KEY      DashScope API Key (qwen/glm 模型用)"
            echo "  DASHSCOPE_BASE_URL     DashScope API Base URL"
            echo "  OPENAI_BASE_URL        OpenAI API 地址 (GPT 模型用)"
            echo "  OPENAI_API_KEY         OpenAI API Key (GPT 模型用)"
            echo "  ANTHROPIC_BASE_URL     Anthropic API 地址 (Claude 模型用)"
            echo "  ANTHROPIC_API_KEY      Anthropic API Key (Claude 模型用)"
            echo "  MODEL                  默认模型"
            echo "  PARALLEL               默认并发数"
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ── 生成输出路径 ──
if [[ -z "$OUTPUT" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT="output/${MODEL}_${TIMESTAMP}.jsonl"
fi

# ── 检查环境 ──
if [[ ! -f ".env" ]] && [[ -z "${DASHSCOPE_API_KEY:-}" ]] && [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "❌ 未找到 .env 文件，且没有设置任何 API Key 环境变量"
    echo "   请创建 .env 文件（参考 .env.example）或设置环境变量后重试"
    exit 1
fi

if [[ ! -f "data/hts_tree.json" ]]; then
    echo "❌ 缺少 HTS 树数据: data/hts_tree.json"
    echo "   请确保数据文件已就位"
    exit 1
fi

# ── 仅评估模式: 直接跳到评估 ──
if [[ -n "$EVAL_ONLY" ]]; then
    if [[ ! -f "$OUTPUT" ]]; then
        echo "❌ 文件不存在: $OUTPUT"
        exit 1
    fi
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  📊 评估模式 (eval-only)"
    echo "  📁 $OUTPUT"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "  [1/2] 精度评估..."
    python eval.py "$OUTPUT"
    echo ""
    echo "  [2/2] CROSS Ruling Database 访问统计..."
    python analyze_cross_usage.py "$OUTPUT"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  ✅ 评估完成"
    echo "═══════════════════════════════════════════════════════════"
    exit 0
fi

# ── 构建推理命令 ──
CMD="python batch_test.py -o \"$OUTPUT\" -m \"$MODEL\" -p $PARALLEL"

if [[ -n "$LIMIT" ]]; then
    CMD="$CMD -n $LIMIT"
fi

if [[ -n "$RESUME" ]]; then
    CMD="$CMD $RESUME"
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# ── 显示配置 ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  🚀 HSCodeComp Agent"
echo "═══════════════════════════════════════════════════════════"
echo "  模型: $MODEL"
echo "  并发: $PARALLEL"
echo "  输出: $OUTPUT"
echo "  命令: $CMD"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── 执行推理 ──
set +e
eval $CMD
EXIT_CODE=$?
set -e

# ── 评估阶段（推理完成后自动执行） ──
if [[ $EXIT_CODE -eq 0 ]] && [[ -f "$OUTPUT" ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  📊 自动评估阶段"
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    # 1. 运行精度评估
    echo "  [1/2] 精度评估..."
    python eval.py "$OUTPUT"
    echo ""

    # 2. 运行 CROSS Ruling Database 访问统计
    echo "  [2/2] CROSS Ruling Database 访问统计..."
    python analyze_cross_usage.py "$OUTPUT"
    echo ""

    echo "═══════════════════════════════════════════════════════════"
    echo "  ✅ 全部完成: $OUTPUT"
    echo "═══════════════════════════════════════════════════════════"
else
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo ""
        echo "❌ 推理阶段异常退出 (exit code: $EXIT_CODE)"
        exit $EXIT_CODE
    fi
fi
