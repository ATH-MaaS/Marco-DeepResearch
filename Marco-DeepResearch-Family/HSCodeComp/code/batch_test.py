#!/usr/bin/env python3
"""
HSCodeComp Agent — 批量自动化测试入口 (hermes-agent)

针对 HSCodeComp benchmark 的 test_data.jsonl 执行批量分类推理，
并把结构化结果写入 output/ 目录。

特性：
  • 支持多模型切换（qwen3.7-max / glm-5.2 / deepseek-v4-pro / gpt-5.5 / claude-opus-4-8）
  • 并发执行 + inline 重试
  • 断点续跑（--resume）
  • 本地 HTS 树工具 + hermes-agent 内置 web 工具
  • GRI 六条归类总规则默认注入

Usage:
    python batch_test.py -o output/result.jsonl
    python batch_test.py -o output/result.jsonl -m qwen3.7-max -p 4
    python batch_test.py -o output/result.jsonl -m claude-opus-4-8 -p 4
    python batch_test.py -o output/result.jsonl --resume
    python batch_test.py -o output/result.jsonl --eval-only
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from _resources import PROJECT_DIR, get_test_data_path

from agent_factory import load_env, load_config, setup_logging, create_agent

# Load environment
load_env()


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_questions(file_path: str) -> List[Dict[str, Any]]:
    """Load test questions from JSONL file."""
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # 使用 task_id 作为 id
                if "id" not in record:
                    record["id"] = str(record.get("task_id", i))
                # 确保有 question 字段
                if "question" not in record:
                    record["question"] = _build_question(record)
                questions.append(record)
            except json.JSONDecodeError:
                continue
    return questions


def _build_question(record: Dict[str, Any]) -> str:
    """Build question from record fields if 'question' field is missing."""
    product_title = record.get("product_name", "")
    product_attributes = record.get("product_attributes", "")
    price = record.get("price", "")
    currency = record.get("currency_code", "")
    lv1 = record.get("cate_lv1_desc", "")
    lv2 = record.get("cate_lv2_desc", "")
    lv3 = record.get("cate_lv3_desc", "")

    return f"""
Given product information, you are required to classify the following product according to US HSCode standards and output a 10-digit HSCode.

You may query the US Customs Rulings website (https://rulings.cbp.gov/home) to obtain HSCode related information.

## Product Information
Product Title: {product_title}
Product Attributes: {product_attributes}
Product Price: {price} {currency}
Product Category: {lv1} -> {lv2} -> {lv3}

## Output Format Requirements
- The final HSCode must be a complete 10-digit code
- Use LaTeX format: `\\boxed{{}}`
- Provide complete classification path and decision rationale
"""


# ═══════════════════════════════════════════════════════════════
# HSCode Extraction
# ═══════════════════════════════════════════════════════════════

def extract_hscode(text: str) -> str:
    """Extract 10-digit HSCode from model response.

    严格对齐官方 eval (test_llm.py) 的 extract_hscode_from_text：
    只从 \\boxed{} 中提取，要求内容长度 >= 10，取第一个匹配项中的纯数字。
    """
    if not text:
        return ""

    # 只从 \boxed{} 提取（与官方 eval 完全一致）
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    matches = [m for m in matches if len(m) >= 10]
    if matches:
        return re.sub(r'[^0-9]', '', str(matches[0]))

    return ""


# ═══════════════════════════════════════════════════════════════
# Thinking Aggregation
# ═══════════════════════════════════════════════════════════════

def aggregate_thinking(messages: List[Dict[str, Any]]) -> str:
    """Extract reasoning/thinking content from messages."""
    chunks: List[str] = []
    for m in messages or []:
        if m.get("role") != "assistant":
            continue
        for key in ("reasoning_content", "reasoning"):
            val = m.get(key)
            if isinstance(val, str) and val.strip():
                chunks.append(val)
    return "\n\n---\n\n".join(chunks)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HSCodeComp Agent — 批量分类测试 (hermes-agent)"
    )
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="输出结果 JSONL 文件路径")
    parser.add_argument("-p", "--parallel", type=int, default=4,
                        help="并发数 (default: 4)")
    parser.add_argument("-m", "--model", type=str, default="qwen3.7-max",
                        help="模型名称 (default: qwen3.7-max)")
    parser.add_argument("-n", "--limit", type=int, default=None,
                        help="限制处理数量（冒烟测试用）")
    parser.add_argument("--resume", action="store_true",
                        help="续跑模式：跳过已成功的 case，重跑失败的")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="单题最大重试次数 (default: 3)")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="单题最大工具调用轮数 (default: 从config读取, 30)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="输入数据文件 (default: HSCodeComp test_data.jsonl)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API Base URL (覆盖 config)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API Key (覆盖 config)")
    parser.add_argument("--enable-thinking", action="store_true", default=None,
                        help="启用 thinking/reasoning")
    parser.add_argument("--no-thinking", action="store_true",
                        help="强制关闭 thinking")
    parser.add_argument("--no-gri-rules", action="store_true",
                        help="不注入 GRI 六条归类总规则（默认注入）")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估已有结果，不执行推理")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="详细日志")
    args = parser.parse_args()

    # Load config
    config = load_config(model_preset=args.model)
    if args.verbose:
        config.setdefault("logging", {})["level"] = "DEBUG"
        config.setdefault("agent", {})["verbose"] = True

    # CLI overrides
    if args.base_url:
        config["base_url"] = args.base_url
    if args.api_key:
        config["api_key"] = args.api_key
    if args.max_iterations:
        config.setdefault("agent", {})["max_iterations"] = args.max_iterations
    if args.no_thinking:
        config["enable_thinking"] = False
    elif args.enable_thinking:
        config["enable_thinking"] = True

    # GRI rules: 默认启用，--no-gri-rules 关闭
    gri_rules = not args.no_gri_rules

    # Setup logging
    logger = setup_logging(config)
    logger.info("═" * 50)
    logger.info("HSCodeComp Agent — 批量分类测试 (hermes-agent)")
    logger.info("═" * 50)

    # Resolve paths
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input) if args.input else get_test_data_path()
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)

    # ── Eval-only mode ──
    if args.eval_only:
        if not output_path.exists():
            print(f"❌ 文件不存在: {output_path}")
            sys.exit(1)
        print(f"\n📊 评估模式: {output_path}")
        # 委托给 eval.py
        import subprocess
        subprocess.run([sys.executable, "eval.py", str(output_path)], check=False)
        sys.exit(0)

    # Resume logic — output 文件只含成功记录，所有 ID 都是已完成
    done_ids: set = set()
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                    done_ids.add(str(rec.get("id", "")))
                except json.JSONDecodeError:
                    continue
        logger.info("♻️  Resume: 跳过 %d 条已成功记录", len(done_ids))

    # Load questions
    questions = load_questions(str(input_path))
    if not questions:
        print("❌ 没有有效的测试数据")
        sys.exit(1)
    if args.limit:
        questions = questions[:args.limit]
    if done_ids:
        questions = [q for q in questions if str(q.get("id", "")) not in done_ids]

    logger.info("📥 输入: %s (%d 题)", input_path, len(questions))
    logger.info("📤 输出: %s", output_path)
    logger.info("🤖 模型: %s (%s)", config.get("model"), config.get("model_preset"))

    if not questions:
        print("✅ 所有题目均已完成")
        sys.exit(0)

    # Batch config
    parallel = args.parallel
    max_retries = args.max_retries
    batch_cfg = config.get("batch", {})
    retry_delay = batch_cfg.get("retry_delay", 2)

    # Test agent creation (也用于 monkey-patch DEFAULT_AGENT_IDENTITY)
    try:
        _test_agent = create_agent(config, logger, gri_rules=gri_rules)
        logger.info("✅ Agent 初始化测试通过")
        del _test_agent
    except Exception as e:
        logger.error("❌ Agent 初始化失败: %s", e, exc_info=True)
        print(f"❌ Agent 初始化失败: {e}")
        sys.exit(1)

    # Prepare output file (non-resume: 不清空已有文件，resume 模式追加)
    if not args.resume:
        output_path.write_text("", encoding="utf-8")

    # Process
    print()
    print("═" * 60)
    print(f"  📋 共 {len(questions)} 个问题待处理")
    print(f"  🤖 模型: {config.get('model')} ({config.get('model_preset')})")
    print(f"  ⚡ 并发: {parallel}")
    print(f"  📤 输出: {output_path}")
    print("═" * 60)
    print()

    results: List[Dict[str, Any]] = []
    total_start = time.time()
    io_lock = threading.Lock()
    total_q = len(questions)

    # 使用 ThreadLocal 存储每线程 agent 实例，避免重复创建
    _thread_local = threading.local()

    def _get_thread_agent() -> Any:
        """获取当前线程的 Agent 实例（每线程创建一次，复用）"""
        if not hasattr(_thread_local, "agent"):
            _thread_local.agent = create_agent(config, logger, gri_rules=gri_rules)
        return _thread_local.agent

    def _process_one(idx: int, question: Dict[str, Any]) -> Dict[str, Any]:
        q_id = str(question.get("id", f"q{idx:04d}"))
        q_text = question.get("question", "")

        with io_lock:
            print(f"[{idx}/{total_q}] ▶ {q_id}: {q_text[:60]}...")

        result = {
            "id": q_id,
            "task_id": question.get("task_id"),
            "product_name": question.get("product_name", ""),
            "hs_code_gt": str(question.get("hs_code", "")),
            "predicted_hscode": "",
            # 兼容 official eval 格式:
            "answer": str(question.get("hs_code", "")),     # ground truth
            "llm_output": "",                                # LLM 完整回复
            "parsed_prediction": "",                         # 提取出的 HSCode
            "status": "error",
            "duration_s": 0.0,
            "model": config.get("model"),
            "model_preset": config.get("model_preset"),
            "thinking": "",
            "error": None,
            "timestamp": datetime.now().isoformat(),
        }

        q_start = time.time()

        for attempt in range(1, max_retries + 1):
            try:
                agent = _get_thread_agent()

                # 记录调用前的 session 累计 token 数（用于计算本次 delta）
                _pre_input = agent.session_input_tokens
                _pre_output = agent.session_output_tokens
                _pre_total = agent.session_total_tokens

                conv_result = agent.run_conversation(q_text)

                answer = conv_result.get("final_response", "") or ""
                predicted = extract_hscode(answer)

                messages = conv_result.get("messages", [])
                thinking = aggregate_thinking(messages)

                result["llm_output"] = answer
                result["predicted_hscode"] = predicted
                result["parsed_prediction"] = predicted
                result["thinking"] = thinking
                result["status"] = "success" if predicted else "no_hscode"
                result["api_calls"] = conv_result.get("api_calls", 0)
                result["iterations"] = conv_result.get("api_calls", 0)

                # Token stats
                result["token_stats"] = {
                    "input_tokens": (conv_result.get("input_tokens", 0) or 0) - _pre_input,
                    "output_tokens": (conv_result.get("output_tokens", 0) or 0) - _pre_output,
                    "total_tokens": (conv_result.get("total_tokens", 0) or 0) - _pre_total,
                }

                if predicted:
                    break  # Success

                if attempt < max_retries:
                    time.sleep(retry_delay)

            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"
                if attempt < max_retries:
                    time.sleep(retry_delay)

        result["duration_s"] = round(time.time() - q_start, 2)

        # Write to file — 只写入成功的结果，失败的不存储方便 resume 重试
        predicted = result.get("predicted_hscode", "")
        with io_lock:
            if predicted:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                gt = result["hs_code_gt"]
                match_2 = predicted[:2] == gt[:2] if len(gt) >= 2 else False
                match_6 = predicted[:6] == gt[:6] if len(gt) >= 6 else False
                match_10 = predicted == gt

                icon = "✅" if match_10 else ("🟡" if match_6 else "❌")
                print(
                    f"  {icon} [{idx}/{total_q}] {q_id} {result['duration_s']}s "
                    f"| pred={predicted} gt={gt} "
                    f"| 2d={'✓' if match_2 else '✗'} 6d={'✓' if match_6 else '✗'} "
                    f"10d={'✓' if match_10 else '✗'}"
                )
            else:
                print(
                    f"  🚫 [{idx}/{total_q}] {q_id} {result['duration_s']}s "
                    f"| 无法提取 HSCode | error={result.get('error', '')[:80]}"
                )

        return result

    # Execute
    if parallel > 1:
        print(f"⚡ 并发模式: max_workers={parallel}\n")
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = [ex.submit(_process_one, i, q) for i, q in enumerate(questions, 1)]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    logger.error("Worker 异常: %s", e, exc_info=True)
    else:
        for idx, q in enumerate(questions, 1):
            results.append(_process_one(idx, q))

    # Summary
    total_elapsed = time.time() - total_start
    total_processed = len(results)
    success = sum(1 for r in results if r.get("predicted_hscode"))

    # Accuracy calculation
    exact_match = 0
    match_2 = 0
    match_4 = 0
    match_6 = 0
    match_8 = 0
    evaluated = 0

    for r in results:
        pred = r.get("predicted_hscode", "")
        gt = str(r.get("hs_code_gt", ""))
        if not pred or not gt or len(gt) < 10:
            continue
        evaluated += 1
        if pred[:2] == gt[:2]:
            match_2 += 1
        if pred[:4] == gt[:4]:
            match_4 += 1
        if pred[:6] == gt[:6]:
            match_6 += 1
        if len(pred) >= 8 and pred[:8] == gt[:8]:
            match_8 += 1
        if pred == gt:
            exact_match += 1

    summary = {
        "model": config.get("model"),
        "model_preset": config.get("model_preset"),
        "input": str(input_path),
        "output": str(output_path),
        "timestamp": datetime.now().isoformat(),
        "total_questions": total_processed,
        "success_extractions": success,
        "evaluated": evaluated,
        "accuracy": {
            "2_digit": round(match_2 / total_processed, 4) if total_processed else 0,
            "4_digit": round(match_4 / total_processed, 4) if total_processed else 0,
            "6_digit": round(match_6 / total_processed, 4) if total_processed else 0,
            "8_digit": round(match_8 / total_processed, 4) if total_processed else 0,
            "10_digit_exact": round(exact_match / total_processed, 4) if total_processed else 0,
        },
        "counts": {
            "2_digit_correct": match_2,
            "4_digit_correct": match_4,
            "6_digit_correct": match_6,
            "8_digit_correct": match_8,
            "10_digit_exact": exact_match,
        },
        "timing": {
            "total_elapsed_s": round(total_elapsed, 2),
            "avg_per_question_s": round(total_elapsed / total_processed, 2) if total_processed else 0,
        },
    }

    # 写入 summary JSON
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print summary
    print()
    print("═" * 60)
    print(f"  📊 批量测试完成")
    print(f"     模型: {config.get('model')} ({config.get('model_preset')})")
    print(f"     总数: {total_processed} | 成功提取: {success} | 评估: {evaluated}")
    print(f"     耗时: {total_elapsed:.1f}s | 平均: {total_elapsed/total_processed:.1f}s/题")
    print()
    if evaluated > 0:
        print(f"  📈 准确率 (分母={total_processed}):")
        print(f"     2-digit:  {match_2}/{total_processed} = {match_2/total_processed*100:.1f}%")
        print(f"     4-digit:  {match_4}/{total_processed} = {match_4/total_processed*100:.1f}%")
        print(f"     6-digit:  {match_6}/{total_processed} = {match_6/total_processed*100:.1f}%")
        print(f"     8-digit:  {match_8}/{total_processed} = {match_8/total_processed*100:.1f}%")
        print(f"     10-digit: {exact_match}/{total_processed} = {exact_match/total_processed*100:.1f}%")
    print()
    print(f"  📤 结果: {output_path}")
    print(f"  📊 汇总: {summary_path}")
    print("═" * 60)

    sys.exit(0 if success > 0 else 2)


if __name__ == "__main__":
    main()
