#!/usr/bin/env python3
"""
HSCode Frontier Test — 交互式运行入口

支持交互式对话和单次分类两种模式。

Usage:
    # 交互式对话（多轮）
    python run.py

    # 单次分类
    python run.py --query "分类这个商品..."

    # 指定模型
    python run.py --model glm-5.2
"""

import argparse
import json
import re
import sys
import time
from typing import Optional

from _resources import PROJECT_DIR

from agent_factory import load_env, load_config, setup_logging, create_agent

# Init environment
load_env()


def extract_hscode(text: str) -> str:
    """Extract 10-digit HSCode from response."""
    if not text:
        return ""
    latex_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(latex_pattern, text)
    for match in matches:
        digits = re.sub(r'[^0-9]', '', match)
        if len(digits) >= 10:
            return digits[:10]
    standalone = re.findall(r'\b(\d{10})\b', text)
    if standalone:
        return standalone[-1]
    return ""


def format_product_query(product_name: str, attributes: str = "",
                         price: str = "", currency: str = "",
                         category: str = "") -> str:
    """Format a product classification query."""
    parts = [
        "Given product information, you are required to classify the following product "
        "according to US HSCode standards and output a 10-digit HSCode.",
        "",
        "## Product Information",
        f"Product Title: {product_name}",
    ]
    if attributes:
        parts.append(f"Product Attributes: {attributes}")
    if price:
        parts.append(f"Product Price: {price} {currency}")
    if category:
        parts.append(f"Product Category: {category}")
    parts.extend([
        "",
        "## Output Format Requirements",
        "- The final HSCode must be a complete 10-digit code",
        "- Use LaTeX format: `\\boxed{}`",
        "- Provide complete classification path and decision rationale",
    ])
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Interactive Loop
# ═══════════════════════════════════════════════════════════════

def run_interactive(agent, logger):
    """Run interactive multi-turn conversation loop."""
    print("\n" + "═" * 60)
    print("  🏷️  HSCode Frontier Test — 交互式商品分类")
    print("  输入商品名称/信息开始分类，输入 /quit 退出")
    print("  输入 /new 清空历史开启新会话")
    print("═" * 60 + "\n")

    turn = 0
    conversation_history: list = []

    while True:
        try:
            user_input = input("👤 商品信息: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("\n👋 再见！")
            break

        if user_input.lower() == "/new":
            conversation_history = []
            turn = 0
            print("\n🔄 已清空对话历史\n")
            continue

        # If input looks like just a product name, wrap it
        if not any(kw in user_input.lower() for kw in ["hscode", "classify", "hsCode", "分类", "归类", "product"]):
            user_input = format_product_query(user_input)

        turn += 1
        logger.info("[Turn %d] 用户: %s", turn, user_input[:100])
        start_time = time.time()

        try:
            result = agent.run_conversation(
                user_input,
                conversation_history=conversation_history,
            )
            elapsed = time.time() - start_time

            conversation_history = result.get("messages", conversation_history)
            response = result.get("final_response", "（无响应）")
            api_calls = result.get("api_calls", 0)

            # Extract HSCode
            hscode = extract_hscode(response)

            logger.info("[Turn %d] 完成 (%.1fs, %d API calls)", turn, elapsed, api_calls)

            print(f"\n🤖 分析结果:\n{response}\n")
            if hscode:
                print(f"  🏷️  提取到 HSCode: {hscode}")
            print(f"  ⏱ {elapsed:.1f}s | 📞 {api_calls} API calls\n")

        except Exception as e:
            logger.error("[Turn %d] 错误: %s", turn, e, exc_info=True)
            print(f"\n❌ 出错了: {e}\n")


# ═══════════════════════════════════════════════════════════════
# Single Query Mode
# ═══════════════════════════════════════════════════════════════

def run_single_query(agent, query: str, logger):
    """Run a single classification query."""
    logger.info("[SingleQuery] %s", query[:100])
    start_time = time.time()

    result = agent.run_conversation(query)
    elapsed = time.time() - start_time

    response = result.get("final_response", "（无响应）")
    hscode = extract_hscode(response)

    print(f"\n{response}\n")
    if hscode:
        print(f"🏷️  HSCode: {hscode}")
    print(f"⏱ {elapsed:.1f}s | 📞 {result.get('api_calls', 0)} API calls")

    return result


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HSCode Frontier Test — 商品分类")
    parser.add_argument("--query", "-q", type=str, help="单次分类：直接输入商品信息")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="模型预设 (qwen3.7-max / glm-5.2)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    args = parser.parse_args()

    # Load config
    config = load_config(model_preset=args.model)
    if args.verbose:
        config.setdefault("logging", {})["level"] = "DEBUG"
        config.setdefault("agent", {})["verbose"] = True

    # Setup
    logger = setup_logging(config)
    logger.info("═" * 40)
    logger.info("🏷️  HSCode Frontier Test 启动 (model=%s)", config.get("model"))
    logger.info("═" * 40)

    # Create agent
    try:
        agent = create_agent(config, logger)
    except Exception as e:
        logger.error("❌ Agent 初始化失败: %s", e, exc_info=True)
        print(f"\n❌ 初始化失败: {e}")
        print("请检查配置文件和环境变量是否正确设置。")
        sys.exit(1)

    # Run
    if args.query:
        run_single_query(agent, args.query, logger)
    else:
        run_interactive(agent, logger)


if __name__ == "__main__":
    main()
