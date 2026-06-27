#!/usr/bin/env python3
"""
HSCode Frontier Test — 评估脚本

对批量测试的结果 JSONL 进行多级精度评估，生成详细的评估报告。

支持：
  • 多级位数精度（2/4/6/8/10 位）
  • 按商品类目分组分析
  • 错误模式归因
  • 多文件横向对比

Usage:
    python eval.py output/hscode_qwen3_7_max_20260626_xxx.jsonl
    python eval.py output/*.jsonl --compare
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_hscode(text: str) -> str:
    """Extract 10-digit HSCode from model response text.

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


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
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


def evaluate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate classification results at multiple digit levels."""
    
    total = len(results)
    evaluated = 0
    
    # Counters
    match_counts = {2: 0, 4: 0, 6: 0, 8: 0, 10: 0}
    
    # Per-category analysis
    category_stats = defaultdict(lambda: {"total": 0, "correct_10": 0, "correct_6": 0})
    
    # Error analysis
    error_levels = Counter()  # At which level does the first mismatch occur
    no_prediction = 0
    
    # Per-item details
    details = []
    
    for r in results:
        # Get predicted and ground truth
        pred = r.get("predicted_hscode", "") or r.get("parsed_prediction", "")
        if not pred:
            # Try to extract from llm_output (our format) or llm_output fallback
            pred = extract_hscode(r.get("llm_output", "") or r.get("answer", ""))
        
        # Ground truth: hs_code_gt (our format) > hs_code (raw data) > answer (official eval format, only if looks like code)
        gt = str(r.get("hs_code_gt", "") or r.get("hs_code", "") or "")
        if not gt:
            # Fallback to 'answer' field only if it looks like a numeric code (official eval format)
            answer_val = str(r.get("answer", ""))
            if answer_val.isdigit():
                gt = answer_val
        
        # Normalize: ensure both are digit strings
        pred = re.sub(r'[^0-9]', '', str(pred))
        gt = re.sub(r'[^0-9]', '', str(gt))
        
        # Pad leading zeros for hs_code that was stored as int (e.g., 0101210010 → int 101210010)
        if gt and len(gt) < 10:
            gt = gt.zfill(10)
        
        if not pred:
            no_prediction += 1
            details.append({
                "id": r.get("id"),
                "product_name": r.get("product_name", "")[:60],
                "predicted": "",
                "ground_truth": gt,
                "error_level": "no_prediction",
            })
            continue
        
        if len(gt) < 10:
            continue  # Skip records without valid ground truth
        
        evaluated += 1
        category = r.get("cate_lv1_desc", "") or "Unknown"
        category_stats[category]["total"] += 1
        
        # Check at each level
        item_detail = {
            "id": r.get("id"),
            "product_name": r.get("product_name", "")[:60],
            "predicted": pred,
            "ground_truth": gt,
            "matches": {},
        }
        
        first_mismatch = None
        for n in [2, 4, 6, 8, 10]:
            if len(pred) >= n and pred[:n] == gt[:n]:
                match_counts[n] += 1
                item_detail["matches"][f"{n}d"] = True
                if n == 10:
                    category_stats[category]["correct_10"] += 1
                if n == 6:
                    category_stats[category]["correct_6"] += 1
            else:
                item_detail["matches"][f"{n}d"] = False
                if first_mismatch is None:
                    first_mismatch = n
        
        if first_mismatch:
            error_levels[f"{first_mismatch}d_mismatch"] += 1
            item_detail["error_level"] = f"{first_mismatch}d"
        else:
            item_detail["error_level"] = "correct"
        
        details.append(item_detail)
    
    # Calculate accuracy — 分母用总记录数（与官方 eval test_llm.py 一致）
    accuracy = {}
    for n in [2, 4, 6, 8, 10]:
        accuracy[f"{n}_digit"] = round(match_counts[n] / total, 4) if total else 0
    
    # Category breakdown
    category_report = {}
    for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        t = stats["total"]
        category_report[cat] = {
            "total": t,
            "accuracy_10d": round(stats["correct_10"] / t, 4) if t else 0,
            "accuracy_6d": round(stats["correct_6"] / t, 4) if t else 0,
        }
    
    return {
        "total_records": total,
        "evaluated": evaluated,
        "no_prediction": no_prediction,
        "accuracy": accuracy,
        "match_counts": match_counts,
        "error_distribution": dict(error_levels.most_common()),
        "category_breakdown": category_report,
        "details": details,
    }


def print_report(eval_result: Dict[str, Any], file_path: str) -> None:
    """Print a human-readable evaluation report."""
    print()
    print("═" * 70)
    print(f"  📊 HSCode Classification Evaluation Report")
    print(f"  📁 File: {Path(file_path).name}")
    print("═" * 70)
    print()
    
    total = eval_result["total_records"]
    evaluated = eval_result["evaluated"]
    no_pred = eval_result["no_prediction"]
    
    print(f"  📋 总记录: {total} | 有效评估: {evaluated} | 无预测: {no_pred}")
    print()
    
    # Accuracy table
    acc = eval_result["accuracy"]
    counts = eval_result["match_counts"]
    print("  ┌──────────────┬──────────┬──────────┐")
    print("  │ Digit Level  │ Accuracy │  Count   │")
    print("  ├──────────────┼──────────┼──────────┤")
    for n in [2, 4, 6, 8, 10]:
        key = f"{n}_digit"
        pct = acc[key] * 100
        cnt = counts[n]
        print(f"  │ {n:>2}-digit      │ {pct:>6.2f}%  │ {cnt:>4}/{evaluated:<4}│")
    print("  └──────────────┴──────────┴──────────┘")
    print()
    
    # Error distribution
    errors = eval_result["error_distribution"]
    if errors:
        print("  📉 Error Distribution (first mismatch level):")
        for level, count in sorted(errors.items()):
            pct = count / evaluated * 100 if evaluated else 0
            print(f"     {level}: {count} ({pct:.1f}%)")
        print()
    
    # Top categories
    cat_report = eval_result["category_breakdown"]
    if cat_report:
        print("  📂 Top Category Performance:")
        print("  ┌────────────────────────────────────┬───────┬─────────┬─────────┐")
        print("  │ Category                           │ Count │ Acc@10d │ Acc@6d  │")
        print("  ├────────────────────────────────────┼───────┼─────────┼─────────┤")
        for cat, stats in list(cat_report.items())[:10]:
            cat_short = cat[:36]
            print(
                f"  │ {cat_short:<36} │ {stats['total']:>5} │ "
                f"{stats['accuracy_10d']*100:>5.1f}%  │ {stats['accuracy_6d']*100:>5.1f}%  │"
            )
        print("  └────────────────────────────────────┴───────┴─────────┴─────────┘")
        print()
    
    print("═" * 70)


def compare_results(file_paths: List[str]) -> None:
    """Compare evaluation results across multiple files."""
    print()
    print("═" * 80)
    print("  📊 HSCode Classification — Multi-Model Comparison")
    print("═" * 80)
    print()
    
    all_evals = []
    for fp in file_paths:
        results = load_results(fp)
        eval_r = evaluate_results(results)
        all_evals.append((Path(fp).stem, eval_r))
    
    # Header
    print("  ┌──────────────────────────────────┬────────┬────────┬────────┬────────┬────────┐")
    print("  │ Model / File                     │  2d    │  4d    │  6d    │  8d    │ 10d    │")
    print("  ├──────────────────────────────────┼────────┼────────┼────────┼────────┼────────┤")
    
    for name, eval_r in all_evals:
        acc = eval_r["accuracy"]
        name_short = name[:34]
        print(
            f"  │ {name_short:<34} │ "
            f"{acc['2_digit']*100:>5.1f}% │ "
            f"{acc['4_digit']*100:>5.1f}% │ "
            f"{acc['6_digit']*100:>5.1f}% │ "
            f"{acc['8_digit']*100:>5.1f}% │ "
            f"{acc['10_digit']*100:>5.1f}% │"
        )
    
    print("  └──────────────────────────────────┴────────┴────────┴────────┴────────┴────────┘")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="HSCode Frontier Test — 评估脚本"
    )
    parser.add_argument(
        "files", nargs="+", type=str,
        help="结果 JSONL 文件路径（支持多个文件）",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="横向对比多个文件的结果",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出评估报告 JSON 文件路径",
    )
    parser.add_argument(
        "--errors-only", action="store_true",
        help="仅输出错误样本详情",
    )
    args = parser.parse_args()
    
    if args.compare and len(args.files) > 1:
        compare_results(args.files)
        return
    
    for fp in args.files:
        if not Path(fp).exists():
            print(f"❌ 文件不存在: {fp}")
            continue
        
        results = load_results(fp)
        if not results:
            print(f"⚠️  文件为空: {fp}")
            continue
        
        eval_result = evaluate_results(results)
        print_report(eval_result, fp)
        
        # Save evaluation JSON
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = Path(fp).with_suffix(".eval.json")
        
        # Don't include full details in output by default (too large)
        output_data = {k: v for k, v in eval_result.items() if k != "details"}
        if args.errors_only:
            output_data["error_details"] = [
                d for d in eval_result["details"]
                if d.get("error_level") != "correct"
            ]
        
        out_path.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  💾 评估报告已保存: {out_path}")
        print()


if __name__ == "__main__":
    main()
