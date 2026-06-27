#!/usr/bin/env python3
"""
HSCodeComp Agent — 批量反思验证入口 (hermes-agent)

对已有推理结果进行反思校验：让同模型基于上一次的完整推理输出，
结合工具重新审核分类结论，验证反思是否带来净正向收益。

特性：
  • 读取已有 output JSONL 作为输入
  • 提供完整 llm_output + 商品信息作为反思上下文
  • 同工具集（HTS + web）可用于验证
  • 输出包含 change_type 用于对比评估
  • 支持并发 + resume

Usage:
    python batch_reflect.py -i output/qwen3.7-max_20260626_120219.jsonl
    python batch_reflect.py -i output/qwen3.7-max_20260626_120219.jsonl -m qwen3.7-max -p 4
    python batch_reflect.py -i output/glm-5.2_20260626_145048.jsonl -p 4
    python batch_reflect.py -i output/result.jsonl --resume
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from _resources import PROJECT_DIR, get_test_data_path

from agent_factory import load_env, load_config, setup_logging, create_agent

# Load environment
load_env()


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_original_results(file_path: str) -> List[Dict[str, Any]]:
    """Load original inference results from JSONL file."""
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                results.append(record)
            except json.JSONDecodeError:
                continue
    return results


def load_test_data_index() -> Dict[str, Dict[str, Any]]:
    """Load test data and build task_id → record index for enriching product info."""
    index = {}
    test_data_path = get_test_data_path()
    if not test_data_path.exists():
        return index
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                tid = str(record.get("task_id", ""))
                if tid:
                    index[tid] = record
            except json.JSONDecodeError:
                continue
    return index


def build_reflect_question(record: Dict[str, Any], test_data: Optional[Dict[str, Any]] = None) -> str:
    """Build the reflection question combining product info + previous analysis."""
    src = test_data or record
    product_title = src.get("product_name", "") or record.get("product_name", "")
    product_attributes = src.get("product_attributes", "")
    price = src.get("price", "")
    currency = src.get("currency_code", "")
    lv1 = src.get("cate_lv1_desc", "")
    lv2 = src.get("cate_lv2_desc", "")
    lv3 = src.get("cate_lv3_desc", "")

    previous_output = record.get("llm_output", "")
    previous_code = record.get("predicted_hscode", "")

    question = f"""请审核以下商品的 HSCode 分类结论。

## 商品信息
Product Title: {product_title}
Product Attributes: {product_attributes}
Product Price: {price} {currency}
Product Category: {lv1} -> {lv2} -> {lv3}

## 上一次的分类分析结论

上一次分析给出的最终 HSCode 为: {previous_code}

以下是完整的分析过程：

---
{previous_output}
---

## 你的任务

请基于上述商品信息和已有分析结论：
1. 审核上一次分类路径中各层级（章/品目/子目）的选择是否正确
2. 使用 HTS 工具验证关键节点，使用网络搜索查找相关判例
3. 如果确认正确，保留原答案；如果发现错误，给出修正后的答案
4. 最终答案使用 LaTeX 格式：`\\boxed{{XXXXXXXXXX}}`
"""
    return question


# ═══════════════════════════════════════════════════════════════
# HSCode Extraction
# ═══════════════════════════════════════════════════════════════

def extract_hscode(text: str) -> str:
    """Extract 10-digit HSCode from model response."""
    if not text:
        return ""
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
# Reflection System Prompt
# ═══════════════════════════════════════════════════════════════

def build_reflect_system_prompt(gri_rules: bool = True) -> str:
    """Build reflection-specific system prompt."""
    prompt_path = PROJECT_DIR / "prompts" / "hscode_reflect.md"
    if prompt_path.exists():
        prompt = prompt_path.read_text(encoding="utf-8")
    else:
        prompt = (
            "你是一位资深的商品编码归类审核专家。"
            "请审核已有的 HSCode 分类结论，验证并确认或修正分类结果。"
            "使用 LaTeX 格式 \\boxed{} 输出最终答案。"
        )

    if gri_rules:
        gri_path = PROJECT_DIR / "prompts" / "gri_rules.md"
        if gri_path.exists():
            prompt += "\n\n" + gri_path.read_text(encoding="utf-8")

    return prompt


# ═══════════════════════════════════════════════════════════════
# Change Type Classification
# ═══════════════════════════════════════════════════════════════

def classify_change(original_pred: str, reflected_pred: str, gt: str) -> str:
    """Classify the change type after reflection."""
    if not reflected_pred:
        return "no_prediction"
    if original_pred == reflected_pred:
        return "unchanged"
    original_correct = (original_pred == gt)
    reflected_correct = (reflected_pred == gt)
    if not original_correct and reflected_correct:
        return "corrected"
    elif original_correct and not reflected_correct:
        return "degraded"
    elif not original_correct and not reflected_correct:
        return "still_wrong"
    else:
        return "unchanged"


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HSCodeComp Agent — 批量反思验证 (hermes-agent)"
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="原始推理结果 JSONL 文件路径（来自 output/ 目录）")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="反思结果输出路径（默认 output_reflect/<model>_reflect_<ts>.jsonl）")
    parser.add_argument("-p", "--parallel", type=int, default=4,
                        help="并发数 (default: 4)")
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="模型名称（默认从输入文件自动检测）")
    parser.add_argument("-n", "--limit", type=int, default=None,
                        help="限制处理数量（冒烟测试用）")
    parser.add_argument("--resume", action="store_true",
                        help="续跑模式：跳过已完成的 case")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="单题最大重试次数 (default: 3)")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="单题最大工具调用轮数 (default: 从config读取)")
    parser.add_argument("--no-gri-rules", action="store_true",
                        help="不注入 GRI 六条归类总规则（默认注入）")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估已有反思结果，不执行推理")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="详细日志")
    args = parser.parse_args()

    # ── Validate input ──
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)

    # Load original results
    original_results = load_original_results(str(input_path))
    if not original_results:
        print(f"❌ 输入文件为空: {input_path}")
        sys.exit(1)

    # Auto-detect model from input
    model_name = args.model
    if not model_name:
        model_name = original_results[0].get("model", "") or "qwen3.7-max"
    print(f"  🤖 检测到模型: {model_name}")

    # ── Determine output path ──
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_DIR / "output_reflect"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{model_name}_reflect_{timestamp}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Eval-only mode ──
    if args.eval_only:
        if not output_path.exists():
            print(f"❌ 文件不存在: {output_path}")
            sys.exit(1)
        print(f"\n📊 评估模式: {output_path}")
        import subprocess
        subprocess.run([sys.executable, "eval_reflect.py", str(output_path)], check=False)
        sys.exit(0)

    # ── Load config ──
    config = load_config(model_preset=model_name)
    if args.verbose:
        config.setdefault("logging", {})["level"] = "DEBUG"
        config.setdefault("agent", {})["verbose"] = True
    if args.max_iterations:
        config.setdefault("agent", {})["max_iterations"] = args.max_iterations

    gri_rules = not args.no_gri_rules

    # Setup logging
    logger = setup_logging(config)
    logger.info("═" * 50)
    logger.info("HSCodeComp Agent — 批量反思验证 (hermes-agent)")
    logger.info("═" * 50)

    # ── Load test data index for enriched product info ──
    test_data_index = load_test_data_index()
    logger.info("📚 加载测试数据索引: %d 条", len(test_data_index))

    # ── Resume logic ──
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
        logger.info("♻️  Resume: 跳过 %d 条已完成记录", len(done_ids))

    # ── Prepare records to process ──
    records = original_results
    if args.limit:
        records = records[:args.limit]
    if done_ids:
        records = [r for r in records if str(r.get("id", "")) not in done_ids]

    logger.info("📥 输入: %s (%d 条原始结果)", input_path, len(original_results))
    logger.info("📤 输出: %s", output_path)
    logger.info("🤖 模型: %s", model_name)
    logger.info("📋 待处理: %d 条", len(records))

    if not records:
        print("✅ 所有记录均已完成反思")
        sys.exit(0)

    # ── Create agent (with reflection system prompt) ──
    reflect_prompt = build_reflect_system_prompt(gri_rules=gri_rules)

    # Monkey-patch build_system_prompt for this session
    import agent_factory
    _original_build = agent_factory.build_system_prompt
    agent_factory.build_system_prompt = lambda gri_rules=True: reflect_prompt

    try:
        _test_agent = create_agent(config, logger, gri_rules=gri_rules)
        logger.info("✅ Reflect Agent 初始化测试通过")
        del _test_agent
    except Exception as e:
        logger.error("❌ Agent 初始化失败: %s", e, exc_info=True)
        print(f"❌ Agent 初始化失败: {e}")
        sys.exit(1)
    finally:
        agent_factory.build_system_prompt = _original_build

    # Prepare output file
    if not args.resume:
        output_path.write_text("", encoding="utf-8")

    # ── Print config ──
    print()
    print("═" * 60)
    print(f"  🔄 HSCode 反思验证")
    print(f"  📥 输入: {input_path.name} ({len(original_results)} 条)")
    print(f"  📋 待处理: {len(records)} 条")
    print(f"  🤖 模型: {model_name}")
    print(f"  ⚡ 并发: {args.parallel}")
    print(f"  📤 输出: {output_path}")
    print("═" * 60)
    print()

    # ── Process ──
    results: List[Dict[str, Any]] = []
    total_start = time.time()
    io_lock = threading.Lock()
    total_q = len(records)

    _thread_local = threading.local()

    def _get_thread_agent() -> Any:
        if not hasattr(_thread_local, "agent"):
            reflect_sys = build_reflect_system_prompt(gri_rules=gri_rules)
            import agent_factory as af
            _orig = af.build_system_prompt
            af.build_system_prompt = lambda gri_rules=True: reflect_sys
            _thread_local.agent = create_agent(config, logger, gri_rules=gri_rules)
            af.build_system_prompt = _orig
        return _thread_local.agent

    max_retries = args.max_retries
    batch_cfg = config.get("batch", {})
    retry_delay = batch_cfg.get("retry_delay", 2)

    def _process_one(idx: int, record: Dict[str, Any]) -> Dict[str, Any]:
        q_id = str(record.get("id", f"q{idx:04d}"))
        task_id = str(record.get("task_id", q_id))
        original_pred = record.get("predicted_hscode", "")
        gt = str(record.get("hs_code_gt", "") or record.get("answer", ""))

        # Build reflection question
        test_data = test_data_index.get(task_id)
        q_text = build_reflect_question(record, test_data)

        with io_lock:
            print(f"[{idx}/{total_q}] ▶ {q_id}: reflecting on pred={original_pred}...")

        result = {
            "id": q_id,
            "task_id": record.get("task_id"),
            "product_name": record.get("product_name", ""),
            "hs_code_gt": gt,
            "answer": gt,
            "original_predicted": original_pred,
            "predicted_hscode": "",
            "reflected_predicted": "",
            "parsed_prediction": "",
            "llm_output": "",
            "change_type": "",
            "status": "error",
            "duration_s": 0.0,
            "model": model_name,
            "thinking": "",
            "error": None,
            "timestamp": datetime.now().isoformat(),
        }

        q_start = time.time()

        for attempt in range(1, max_retries + 1):
            try:
                agent = _get_thread_agent()

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
                result["reflected_predicted"] = predicted
                result["parsed_prediction"] = predicted
                result["thinking"] = thinking
                result["status"] = "success" if predicted else "no_hscode"
                result["iterations"] = conv_result.get("api_calls", 0)

                # Token stats
                result["token_stats"] = {
                    "input_tokens": (conv_result.get("input_tokens", 0) or 0) - _pre_input,
                    "output_tokens": (conv_result.get("output_tokens", 0) or 0) - _pre_output,
                    "total_tokens": (conv_result.get("total_tokens", 0) or 0) - _pre_total,
                }

                if predicted:
                    break

                if attempt < max_retries:
                    time.sleep(retry_delay)

            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"
                if attempt < max_retries:
                    time.sleep(retry_delay)

        result["duration_s"] = round(time.time() - q_start, 2)

        # Classify change type
        reflected_pred = result.get("predicted_hscode", "")
        result["change_type"] = classify_change(original_pred, reflected_pred, gt)

        # Write to file
        with io_lock:
            if reflected_pred:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                change = result["change_type"]
                icon_map = {
                    "corrected": "🟢",
                    "degraded": "🔴",
                    "unchanged": "⚪",
                    "still_wrong": "🟡",
                    "no_prediction": "🚫",
                }
                icon = icon_map.get(change, "❓")
                match_10 = reflected_pred == gt
                print(
                    f"  {icon} [{idx}/{total_q}] {q_id} {result['duration_s']}s "
                    f"| {change} | orig={original_pred} → refl={reflected_pred} gt={gt} "
                    f"| exact={'✓' if match_10 else '✗'}"
                )
            else:
                print(
                    f"  🚫 [{idx}/{total_q}] {q_id} {result['duration_s']}s "
                    f"| 无法提取 HSCode | error={result.get('error', '')[:80]}"
                )

        return result

    # Execute
    if args.parallel > 1:
        print(f"⚡ 并发模式: max_workers={args.parallel}\n")
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = [ex.submit(_process_one, i, r) for i, r in enumerate(records, 1)]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    logger.error("Worker 异常: %s", e, exc_info=True)
    else:
        for idx, r in enumerate(records, 1):
            results.append(_process_one(idx, r))

    # ── Summary ──
    total_elapsed = time.time() - total_start
    total_processed = len(results)
    success = sum(1 for r in results if r.get("predicted_hscode"))

    change_counts = {}
    for r in results:
        ct = r.get("change_type", "unknown")
        change_counts[ct] = change_counts.get(ct, 0) + 1

    corrected = change_counts.get("corrected", 0)
    degraded = change_counts.get("degraded", 0)
    unchanged = change_counts.get("unchanged", 0)
    still_wrong = change_counts.get("still_wrong", 0)
    net_gain = corrected - degraded

    summary = {
        "model": model_name,
        "input": str(input_path),
        "output": str(output_path),
        "timestamp": datetime.now().isoformat(),
        "total_records": total_processed,
        "success_extractions": success,
        "change_types": change_counts,
        "net_gain": net_gain,
        "timing": {
            "total_elapsed_s": round(total_elapsed, 2),
            "avg_per_question_s": round(total_elapsed / total_processed, 2) if total_processed else 0,
        },
    }

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("═" * 60)
    print(f"  📊 反思验证完成")
    print(f"     模型: {model_name}")
    print(f"     总数: {total_processed} | 成功提取: {success}")
    print(f"     耗时: {total_elapsed:.1f}s | 平均: {total_elapsed/total_processed:.1f}s/题")
    print()
    print(f"  🔄 反思效果:")
    print(f"     🟢 corrected (错→对): {corrected}")
    print(f"     🔴 degraded  (对→错): {degraded}")
    print(f"     ⚪ unchanged (未变化): {unchanged}")
    print(f"     🟡 still_wrong (仍错): {still_wrong}")
    print(f"     📈 Net gain: {net_gain:+d}")
    print()
    print(f"  📤 结果: {output_path}")
    print(f"  📊 汇总: {summary_path}")
    print("═" * 60)

    sys.exit(0 if success > 0 else 2)


if __name__ == "__main__":
    main()
