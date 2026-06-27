#!/usr/bin/env python3
"""
HSCode Frontier Test — 反思效果评估脚本

对比反思前后的分类结果，统计：
  • correct→correct (stable_correct): 原来对，反思后仍对
  • wrong→correct (corrected): 原来错，反思后改对了
  • correct→wrong (degraded): 原来对，反思后改错了
  • wrong→wrong (still_wrong): 原来错，反思后仍然错
  • Net gain = corrected - degraded

Usage:
    python eval_reflect.py output_reflect/qwen3.7-max_reflect_20260627.jsonl
    python eval_reflect.py output_reflect/*.jsonl --compare
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def normalize_code(code: str) -> str:
    """Normalize HS code: extract digits only, pad to 10."""
    code = re.sub(r'[^0-9]', '', str(code or ""))
    if code and len(code) < 10:
        code = code.zfill(10)
    return code


def evaluate_reflection(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate reflection results at multiple digit levels.

    Returns detailed metrics showing how reflection affected accuracy.
    """
    total = len(results)

    # Per-level analysis (2, 4, 6, 8, 10 digit)
    level_stats = {}
    for n in [2, 4, 6, 8, 10]:
        level_stats[n] = {
            "stable_correct": 0,   # was correct, still correct
            "corrected": 0,        # was wrong, now correct
            "degraded": 0,         # was correct, now wrong
            "still_wrong": 0,      # was wrong, still wrong
            "no_prediction": 0,    # no reflected prediction
        }

    # Overall 10-digit stats (primary metric)
    change_types = Counter()
    details = []

    for r in results:
        original_pred = normalize_code(r.get("original_predicted", ""))
        reflected_pred = normalize_code(r.get("predicted_hscode", "") or r.get("reflected_predicted", ""))
        gt = normalize_code(r.get("hs_code_gt", "") or r.get("answer", ""))

        if not gt or len(gt) < 10:
            continue

        # Record the change_type from batch_reflect (10-digit level)
        ct = r.get("change_type", "")
        if ct:
            change_types[ct] += 1

        detail = {
            "id": r.get("id"),
            "product_name": r.get("product_name", "")[:50],
            "original": original_pred,
            "reflected": reflected_pred,
            "gt": gt,
            "changes": {},
        }

        # Per-level analysis
        for n in [2, 4, 6, 8, 10]:
            if not reflected_pred or len(reflected_pred) < n:
                level_stats[n]["no_prediction"] += 1
                detail["changes"][f"{n}d"] = "no_pred"
                continue

            orig_match = (len(original_pred) >= n and original_pred[:n] == gt[:n])
            refl_match = (reflected_pred[:n] == gt[:n])

            if orig_match and refl_match:
                level_stats[n]["stable_correct"] += 1
                detail["changes"][f"{n}d"] = "stable_correct"
            elif not orig_match and refl_match:
                level_stats[n]["corrected"] += 1
                detail["changes"][f"{n}d"] = "corrected"
            elif orig_match and not refl_match:
                level_stats[n]["degraded"] += 1
                detail["changes"][f"{n}d"] = "degraded"
            else:
                level_stats[n]["still_wrong"] += 1
                detail["changes"][f"{n}d"] = "still_wrong"

        details.append(detail)

    # Calculate accuracy before/after
    evaluated = len(details)
    accuracy_before = {}
    accuracy_after = {}
    net_gains = {}

    for n in [2, 4, 6, 8, 10]:
        stats = level_stats[n]
        before_correct = stats["stable_correct"] + stats["degraded"]
        after_correct = stats["stable_correct"] + stats["corrected"]
        accuracy_before[f"{n}_digit"] = round(before_correct / total, 4) if total else 0
        accuracy_after[f"{n}_digit"] = round(after_correct / total, 4) if total else 0
        net_gains[f"{n}_digit"] = stats["corrected"] - stats["degraded"]

    return {
        "total_records": total,
        "evaluated": evaluated,
        "change_types_10d": dict(change_types),
        "level_stats": {f"{n}d": level_stats[n] for n in [2, 4, 6, 8, 10]},
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "net_gains": net_gains,
        "details": details,
    }


def print_report(eval_result: Dict[str, Any], file_path: str) -> None:
    """Print a human-readable reflection evaluation report."""
    print()
    print("═" * 70)
    print(f"  🔄 HSCode Reflection Evaluation Report")
    print(f"  📁 File: {Path(file_path).name}")
    print("═" * 70)
    print()

    total = eval_result["total_records"]
    evaluated = eval_result["evaluated"]
    print(f"  📋 总记录: {total} | 有效评估: {evaluated}")
    print()

    # Change type summary (10-digit)
    ct = eval_result["change_types_10d"]
    corrected = ct.get("corrected", 0)
    degraded = ct.get("degraded", 0)
    unchanged = ct.get("unchanged", 0)
    still_wrong = ct.get("still_wrong", 0)
    net = corrected - degraded

    print("  🎯 10-digit 反思效果总结:")
    print(f"     🟢 corrected (错→对): {corrected:>4}")
    print(f"     🔴 degraded  (对→错): {degraded:>4}")
    print(f"     ⚪ unchanged (未变化): {unchanged:>4}")
    print(f"     🟡 still_wrong (仍错): {still_wrong:>4}")
    print(f"     {'📈' if net >= 0 else '📉'} Net gain: {net:+d}")
    print()

    # Per-level table
    print("  ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    print("  │  Level   │ Acc(前)  │ Acc(后)  │ Δ Acc    │ Corrected│ Degraded │ Net Gain │")
    print("  ├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

    for n in [2, 4, 6, 8, 10]:
        key = f"{n}_digit"
        before = eval_result["accuracy_before"][key] * 100
        after = eval_result["accuracy_after"][key] * 100
        delta = after - before
        stats = eval_result["level_stats"][f"{n}d"]
        corr = stats["corrected"]
        degr = stats["degraded"]
        gain = corr - degr
        delta_str = f"{delta:+.2f}%"
        gain_str = f"{gain:+d}"
        print(
            f"  │ {n:>2}-digit  │ {before:>6.2f}%  │ {after:>6.2f}%  │ {delta_str:>8} │ {corr:>8} │ {degr:>8} │ {gain_str:>8} │"
        )

    print("  └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
    print()

    # Show specific corrections and degradations (10-digit)
    details = eval_result.get("details", [])
    corrected_cases = [d for d in details if d["changes"].get("10d") == "corrected"]
    degraded_cases = [d for d in details if d["changes"].get("10d") == "degraded"]

    if corrected_cases:
        print(f"  🟢 Corrected cases (错→对, top 10):")
        for d in corrected_cases[:10]:
            print(f"     {d['id']:>6} | {d['product_name']:<40} | {d['original']} → {d['reflected']} (gt={d['gt']})")
        if len(corrected_cases) > 10:
            print(f"     ... and {len(corrected_cases) - 10} more")
        print()

    if degraded_cases:
        print(f"  🔴 Degraded cases (对→错, top 10):")
        for d in degraded_cases[:10]:
            print(f"     {d['id']:>6} | {d['product_name']:<40} | {d['original']} → {d['reflected']} (gt={d['gt']})")
        if len(degraded_cases) > 10:
            print(f"     ... and {len(degraded_cases) - 10} more")
        print()

    print("═" * 70)


def compare_results(file_paths: List[str]) -> None:
    """Compare reflection results across multiple files."""
    print()
    print("═" * 80)
    print("  🔄 HSCode Reflection — Multi-Model Comparison")
    print("═" * 80)
    print()

    all_evals = []
    for fp in file_paths:
        results = load_results(fp)
        eval_r = evaluate_reflection(results)
        all_evals.append((Path(fp).stem, eval_r))

    print("  ┌──────────────────────────────────┬────────┬────────┬────────┬────────┬────────┐")
    print("  │ Model / File                     │ Corr'd │ Degrad │  Net   │Acc(前) │Acc(後) │")
    print("  ├──────────────────────────────────┼────────┼────────┼────────┼────────┼────────┤")

    for name, eval_r in all_evals:
        ct = eval_r["change_types_10d"]
        corr = ct.get("corrected", 0)
        degr = ct.get("degraded", 0)
        net = corr - degr
        before = eval_r["accuracy_before"]["10_digit"] * 100
        after = eval_r["accuracy_after"]["10_digit"] * 100
        name_short = name[:34]
        print(
            f"  │ {name_short:<34} │ {corr:>5}  │ {degr:>5}  │ {net:>+5}  │ {before:>5.1f}% │ {after:>5.1f}% │"
        )

    print("  └──────────────────────────────────┴────────┴────────┴────────┴────────┴────────┘")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="HSCode Frontier Test — 反思效果评估"
    )
    parser.add_argument(
        "files", nargs="+", type=str,
        help="反思结果 JSONL 文件路径（支持多个文件）",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="横向对比多个文件的反思效果",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出评估报告 JSON 文件路径",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="显示所有变化案例（不截断）",
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

        eval_result = evaluate_reflection(results)
        print_report(eval_result, fp)

        # Save evaluation JSON
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = Path(fp).with_suffix(".reflect_eval.json")

        # Don't include full details in output by default
        output_data = {k: v for k, v in eval_result.items() if k != "details"}
        if args.show_all:
            output_data["details"] = eval_result["details"]

        out_path.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  💾 评估报告已保存: {out_path}")
        print()


if __name__ == "__main__":
    main()
