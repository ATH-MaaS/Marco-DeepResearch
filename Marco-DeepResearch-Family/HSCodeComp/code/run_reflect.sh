#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# HSCode Frontier Test — 反思验证一键脚本
# ═══════════════════════════════════════════════════════════════
# Usage:
#   ./run_reflect.sh -i output/qwen3.7-max_20260626_120219.jsonl
#   ./run_reflect.sh -i output/qwen3.7-max_20260626_120219.jsonl -m qwen3.7-max -p 4
#   ./run_reflect.sh -i output/glm-5.2_20260626_145048.jsonl -m glm-5.2 -p 4
#   ./run_reflect.sh -i output/result.jsonl -n 10      # 冒烟测试(10题)
#   ./run_reflect.sh -i output/result.jsonl --resume   # 续跑
#   ./run_reflect.sh --eval-only -o output_reflect/xxx.jsonl  # 仅评估

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 默认参数 ──
INPUT=""
MODEL=""
PARALLEL="${PARALLEL:-4}"
LIMIT=""
RESUME=""
EVAL_ONLY=""
OUTPUT=""
EXTRA_ARGS=""

# ── 解析参数 ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)     INPUT="$2"; shift 2 ;;
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
            echo "对已有推理结果进行反思校验，验证反思是否带来净正向收益。"
            echo ""
            echo "Options:"
            echo "  -i, --input PATH       原始推理结果 JSONL 文件 (必选)"
            echo "  -m, --model MODEL      模型名称 (默认: 从输入文件自动检测)"
            echo "  -p, --parallel N       并发数 (default: 4)"
            echo "  -n, --limit N          限制题数 (冒烟测试)"
            echo "  -o, --output PATH      输出路径 (default: output_reflect/<model>_reflect_<ts>.jsonl)"
            echo "  --resume               续跑模式 (跳过已完成 case)"
            echo "  --eval-only            仅评估已有反思结果"
            echo "  -v, --verbose          详细日志"
            echo "  -h, --help             显示帮助"
            echo ""
            echo "Examples:"
            echo "  $0 -i output/qwen3.7-max_20260626_120219.jsonl -p 4"
            echo "  $0 -i output/glm-5.2_20260626_145048.jsonl -m glm-5.2"
            echo "  $0 --eval-only -o output_reflect/qwen3.7-max_reflect_20260627.jsonl"
            echo ""
            echo "Environment Variables:"
            echo "  DASHSCOPE_API_KEY      DashScope API Key (qwen/glm 模型用)"
            echo "  DASHSCOPE_BASE_URL     DashScope API Base URL"
            echo "  OPENAI_BASE_URL        OpenAI API 地址 (GPT 模型用)"
            echo "  OPENAI_API_KEY         OpenAI API Key (GPT 模型用)"
            echo "  ANTHROPIC_BASE_URL     Anthropic API 地址 (Claude 模型用)"
            echo "  ANTHROPIC_API_KEY      Anthropic API Key (Claude 模型用)"
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ── 仅评估模式 ──
if [[ -n "$EVAL_ONLY" ]]; then
    if [[ -z "$OUTPUT" ]]; then
        echo "❌ --eval-only 模式需要指定 -o 参数"
        exit 1
    fi
    if [[ ! -f "$OUTPUT" ]]; then
        echo "❌ 文件不存在: $OUTPUT"
        exit 1
    fi
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  📊 反思评估模式 (eval-only)"
    echo "  📁 $OUTPUT"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    python eval_reflect.py "$OUTPUT"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  ✅ 评估完成"
    echo "═══════════════════════════════════════════════════════════"
    exit 0
fi

# ── 检查必选参数 ──
if [[ -z "$INPUT" ]]; then
    echo "❌ 缺少输入文件参数 -i/--input"
    echo "   Usage: $0 -i output/model_timestamp.jsonl"
    exit 1
fi

if [[ ! -f "$INPUT" ]]; then
    echo "❌ 输入文件不存在: $INPUT"
    exit 1
fi

# ── 检查环境 ──
if [[ ! -f "data/hts_tree.json" ]]; then
    echo "❌ 缺少 HTS 树数据: data/hts_tree.json"
    exit 1
fi

# ── 生成输出路径 ──
if [[ -z "$OUTPUT" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    # 从输入文件名推断模型名（如果未指定）
    if [[ -z "$MODEL" ]]; then
        # 从文件名提取模型（如 qwen3.7-max_20260626_120219.jsonl → qwen3.7-max）
        BASENAME=$(basename "$INPUT" .jsonl)
        MODEL=$(echo "$BASENAME" | sed 's/_[0-9]\{8\}_[0-9]\{6\}$//')
    fi
    mkdir -p output_reflect
    OUTPUT="output_reflect/${MODEL}_reflect_${TIMESTAMP}.jsonl"
fi

# ── 构建推理命令 ──
CMD="python batch_reflect.py -i \"$INPUT\" -o \"$OUTPUT\" -p $PARALLEL"

if [[ -n "$MODEL" ]]; then
    CMD="$CMD -m \"$MODEL\""
fi

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
echo "  🔄 HSCode Reflection Verification"
echo "═══════════════════════════════════════════════════════════"
echo "  输入: $INPUT"
echo "  模型: ${MODEL:-auto-detect}"
echo "  并发: $PARALLEL"
echo "  输出: $OUTPUT"
echo "  命令: $CMD"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── 执行反思推理 ──
set +e
eval $CMD
EXIT_CODE=$?
set -e

# ── 评估阶段（反思完成后自动执行） ──
if [[ $EXIT_CODE -eq 0 ]] && [[ -f "$OUTPUT" ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  📊 自动评估阶段"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    python eval_reflect.py "$OUTPUT"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  ✅ 全部完成: $OUTPUT"
    echo "═══════════════════════════════════════════════════════════"
else
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo ""
        echo "❌ 反思推理阶段异常退出 (exit code: $EXIT_CODE)"
        exit $EXIT_CODE
    fi
fi
