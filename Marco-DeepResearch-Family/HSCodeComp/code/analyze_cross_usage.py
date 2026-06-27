#!/usr/bin/env python3
"""
统计 CROSS Ruling Database (rulings.cbp.gov) 的访问次数和比例。

通过扫描输出文件中的 llm_output / messages 字段，检测每条测试样例
在解题过程中是否实际访问了 CBP CROSS ruling database。

检测策略（按精确度递减）：
  1. URL 访问: llm_output 中包含 "rulings.cbp.gov" → 实际通过 web 工具访问了
  2. 文字提及: llm_output 中提及 "ruling"/"cbp" 但无 URL → 可能是模型知识回忆

Usage:
    python analyze_cross_usage.py output/result.jsonl
    python analyze_cross_usage.py ../hscode_benchmark/benchmark/benchmark_final_testset_agent/*.json
    python analyze_cross_usage.py output/*.jsonl --detail
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def analyze_file(file_path: str, show_detail: bool = False) -> Dict[str, Any]:
    """Analyze a single result file for CROSS ruling database usage."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return {}

    # Load records (support both JSON array and JSONL)
    records = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        else:
            print(f"⚠️  不支持的格式: {file_path}")
            return {}

    if not records:
        print(f"⚠️  文件为空: {file_path}")
        return {}

    # Analyze each record
    total = len(records)
    url_accessed = []  # 实际访问 rulings.cbp.gov
    text_mentions = []  # 仅文字提及
    no_mention = []  # 完全未涉及

    for i, rec in enumerate(records):
        # Combine all text sources for detection
        text_sources = []
        text_sources.append(rec.get("llm_output", "") or "")
        text_sources.append(rec.get("answer", "") or "" if not str(rec.get("answer", "")).isdigit() else "")

        # If messages field exists (hermes-agent style), scan tool results
        messages = rec.get("messages", [])
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    text_sources.append(content)

        combined = "\n".join(text_sources)
        combined_lower = combined.lower()

        # Detection
        has_url = "rulings.cbp.gov" in combined
        has_mention = (
            "ruling" in combined_lower
            or "cbp" in combined_lower
            or "customs ruling" in combined_lower
            or "cross database" in combined_lower
        )

        record_id = rec.get("id", rec.get("task_id", rec.get("itemid", i)))

        if has_url:
            url_accessed.append(record_id)
        elif has_mention:
            text_mentions.append(record_id)
        else:
            no_mention.append(record_id)

    # Report
    result = {
        "file": path.name,
        "total": total,
        "url_accessed": len(url_accessed),
        "url_accessed_pct": round(len(url_accessed) / total * 100, 1),
        "text_mentions": len(text_mentions),
        "text_mentions_pct": round(len(text_mentions) / total * 100, 1),
        "no_mention": len(no_mention),
        "no_mention_pct": round(len(no_mention) / total * 100, 1),
    }

    print(f"\n{'═' * 60}")
    print(f"  📊 CROSS Ruling Database 使用统计")
    print(f"  📁 {path.name}")
    print(f"{'═' * 60}")
    print(f"  总样例数: {total}")
    print()
    print(f"  🌐 实际访问 rulings.cbp.gov:  {len(url_accessed):>4} ({result['url_accessed_pct']:.1f}%)")
    print(f"  📝 仅文字提及 ruling/CBP:     {len(text_mentions):>4} ({result['text_mentions_pct']:.1f}%)")
    print(f"  ⬜ 完全未涉及:                {len(no_mention):>4} ({result['no_mention_pct']:.1f}%)")
    print(f"{'═' * 60}")

    if show_detail and url_accessed:
        print(f"\n  实际访问 CROSS 的样例 ID (前20):")
        for rid in url_accessed[:20]:
            print(f"    - {rid}")
        if len(url_accessed) > 20:
            print(f"    ... 共 {len(url_accessed)} 条")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="统计 CROSS Ruling Database 访问次数和比例"
    )
    parser.add_argument("files", nargs="+", help="结果文件路径 (JSONL 或 JSON)")
    parser.add_argument("--detail", action="store_true", help="显示访问 CROSS 的样例 ID")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出统计 JSON")
    args = parser.parse_args()

    all_results = []
    for fp in args.files:
        result = analyze_file(fp, show_detail=args.detail)
        if result:
            all_results.append(result)

    # Comparison table if multiple files
    if len(all_results) > 1:
        print(f"\n{'═' * 70}")
        print(f"  📊 横向对比")
        print(f"{'═' * 70}")
        print(f"  {'File':<35} {'Total':>5} {'URL访问':>8} {'文字提及':>8} {'未涉及':>8}")
        print(f"  {'─' * 35} {'─' * 5} {'─' * 8} {'─' * 8} {'─' * 8}")
        for r in all_results:
            print(
                f"  {r['file']:<35} {r['total']:>5} "
                f"{r['url_accessed_pct']:>6.1f}% "
                f"{r['text_mentions_pct']:>6.1f}% "
                f"{r['no_mention_pct']:>6.1f}%"
            )
        print(f"{'═' * 70}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  💾 统计结果已保存: {args.output}")


if __name__ == "__main__":
    main()
