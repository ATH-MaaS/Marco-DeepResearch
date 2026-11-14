"""Unified two-phase pipeline: collect skills on train set, evaluate on test set.

Two data loading strategies:
  (a) Explicit files: --train_file + --test_file
  (b) Split file:     --data_path + --split_file

Phase 1 – Skill Collection (train set):
    Run search_skill on train split.
    Skills accumulate in a persistent library across questions.

Phase 2 – Evaluation (test set):
    Run multiple modes on test split. All eval logic is reused from eval_benchmark.py.

Usage:
    # With explicit train/test files
    python scripts/eval_pipeline.py --benchmark supergpqa_hard_mathematics \
        --train_file data/supergpqa_reuse_dataset/hard_mathematics/hard_mathematics_train.json \
        --test_file data/supergpqa_reuse_dataset/hard_mathematics/hard_mathematics_test.json \
        --model qwen3-8b --test_mode direct code search_skill skill_only

    # Skip training, use an existing skill library
    python scripts/eval_pipeline.py --benchmark supergpqa_hard_mathematics \
        --test_file data/.../test.json \
        --skip_train --skill_library results/eval/.../skill_library.json \
        --model qwen3-8b --test_mode skill_only
"""

import argparse
import asyncio
import functools
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from typing import Dict, List

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.CRITICAL, force=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.client import LLMClient
from benchmarks import load_benchmark
from evaluation.usage_tracker import attach_usage_tracking
from core.agent import get_page_visit_stats

import importlib.util

_EVAL_BENCHMARK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_benchmark.py")
_eval_benchmark_spec = importlib.util.spec_from_file_location("eval_benchmark", _EVAL_BENCHMARK_PATH)
_eval_benchmark = importlib.util.module_from_spec(_eval_benchmark_spec)
_eval_benchmark_spec.loader.exec_module(_eval_benchmark)

load_config = _eval_benchmark.load_config
RealtimeWriter = _eval_benchmark.RealtimeWriter
_agg = _eval_benchmark._agg
_group_by = _eval_benchmark._group_by
_fmt_pct = _eval_benchmark._fmt_pct
_ascii_table = _eval_benchmark._ascii_table
_async_evaluate = _eval_benchmark._async_evaluate
eval_direct = _eval_benchmark.eval_direct
eval_code = _eval_benchmark.eval_code
eval_search_code = _eval_benchmark.eval_search_code
eval_free_tool = _eval_benchmark.eval_free_tool
eval_search_skill = _eval_benchmark.eval_search_skill
eval_search_skill_no_code = _eval_benchmark.eval_search_skill_no_code
eval_skill_only = _eval_benchmark.eval_skill_only
eval_skill_only_no_code = _eval_benchmark.eval_skill_only_no_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-phase pipeline: train (collect skills) → test (evaluate)")

    # Benchmark
    p.add_argument("--benchmark", type=str, required=True,
                   help="Benchmark name (evoagentbench, livecode_bench, etc.)")

    # Data loading strategy A: explicit train/test files
    p.add_argument("--train_file", type=str, default=None,
                   help="Training data path (JSONL/JSON)")
    p.add_argument("--test_file", type=str, default=None,
                   help="Test data path (JSONL/JSON)")

    # Data loading strategy B: single dataset + split file
    p.add_argument("--data_path", type=str, default=None,
                   help="Dataset path (used with --split_file)")
    p.add_argument("--split_file", type=str, default=None,
                   help="JSON file defining train/test split by IDs")

    # Model & config
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    # Train phase
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training phase; use --skill_library directly")
    p.add_argument("--skip_test", action="store_true",
                   help="Skip test phase; only run skill collection on train set")
    p.add_argument("--train_num", type=int, default=-1,
                   help="Number of train questions (-1 = all)")
    p.add_argument("--train_mode", type=str, default="search_skill",
                   choices=["search_skill", "search_skill_no_code",
                            "search_skill_force"],
                   help="Mode for skill collection on train set")

    # Test phase
    ALL_MODES = ["direct", "code", "search_code", "free_tool", "search_skill",
                 "search_skill_no_code", "skill_only", "skill_only_no_code"]
    p.add_argument("--test_mode", type=str, nargs="+",
                   default=["direct", "code", "free_tool", "search_skill", "skill_only"],
                   choices=ALL_MODES,
                   help="Test evaluation mode(s).")
    p.add_argument("--test_num", type=int, default=-1,
                   help="Number of test questions (-1 = all)")

    # Agent settings
    p.add_argument("--max_rounds", type=int, default=15)
    p.add_argument("--concurrency", type=int, default=20)

    # Skill settings
    p.add_argument("--skill_library", type=str, default=None,
                   help="Path to pre-built skill library")
    p.add_argument("--skill_top_k", type=int, default=None)

    # Output
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--resume", action="store_true")

    # Judge
    p.add_argument("--judge_profile", type=str, default=None)

    # Subset filter
    p.add_argument("--subset", type=str, default=None,
                   help="Filter by subset (difficulty/domain)")

    p.add_argument("--repeat_avg", action="store_true",
                   help="Run each test mode 3 times and report average accuracy")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_items_from_file(benchmark, file_path, num_samples=-1, subset=None, seed=42):
    """Load items from a file path."""
    if file_path.endswith(".jsonl"):
        items = benchmark.load(file_path, num_samples=-1, subset=subset, seed=seed)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        if subset:
            items = [it for it in items if it.get("domain", "").lower() == subset.lower()]
    if num_samples > 0 and num_samples < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, num_samples)
    return items


def load_items_from_split(benchmark, data_path, split_file, split, num_samples=-1,
                          subset=None, seed=42):
    """Load items filtered by a split file (train/test IDs)."""
    with open(split_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    split_ids = set()
    for cluster in raw.get("clusters", {}).values():
        split_ids.update(str(x) for x in cluster.get(split, []))

    all_items = benchmark.load(data_path or "", num_samples=-1, subset=subset, seed=seed)
    items = [it for it in all_items if str(it["id"]) in split_ids]

    if num_samples > 0 and num_samples < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, num_samples)
    return items


def load_train_items(benchmark, args):
    """Load training items based on CLI args."""
    if args.train_file:
        return load_items_from_file(
            benchmark, args.train_file, args.train_num, args.subset, args.seed)
    elif args.split_file and args.data_path:
        return load_items_from_split(
            benchmark, args.data_path, args.split_file, "train",
            args.train_num, args.subset, args.seed)
    else:
        print("Error: Must provide either --train_file or (--data_path + --split_file)")
        sys.exit(1)


def load_test_items(benchmark, args):
    """Load test items based on CLI args."""
    if args.test_file:
        return load_items_from_file(
            benchmark, args.test_file, args.test_num, args.subset, args.seed)
    elif args.split_file and args.data_path:
        return load_items_from_split(
            benchmark, args.data_path, args.split_file, "test",
            args.test_num, args.subset, args.seed)
    else:
        print("Error: Must provide either --test_file or (--data_path + --split_file)")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Phase 1: Skill collection on train set
# ---------------------------------------------------------------------------

async def run_train_phase(items, benchmark, config, model_profile, args,
                          skill_library_path, output_dir):
    """Run search_skill on train set to collect skills."""
    skill_cfg = config.get("skill", {})
    skill_top_k = args.skill_top_k or skill_cfg.get("skill_top_k", 3)
    retrieval_method = skill_cfg.get("retrieval", "embedding")

    rt_writer = RealtimeWriter(output_dir, f"train_{args.train_mode}")
    rt_writer.prepare(resume=args.resume)

    if args.resume:
        completed_ids = rt_writer.get_completed_ids()
        run_items = [it for it in items if it["id"] not in completed_ids]
        if completed_ids:
            print(f"  [Resume] Found {len(completed_ids)} completed, "
                  f"{len(run_items)} remaining (of {len(items)} total)")
        if not run_items:
            print(f"  [Resume] All train items completed, skipping.")
            return
    else:
        run_items = items

    if args.train_mode == "search_skill_no_code":
        results = await eval_search_skill_no_code(
            run_items, benchmark, config, model_profile,
            args.concurrency, args.max_rounds,
            skill_library_path, skill_top_k,
            retrieval=retrieval_method,
            batch_size=args.concurrency,
            rt_writer=rt_writer,
            resume=args.resume,
            disable_retrieval=True,
        )
    elif args.train_mode == "search_skill_force":
        results = await eval_search_skill(
            run_items, benchmark, config, model_profile,
            args.concurrency, args.max_rounds,
            skill_library_path, skill_top_k,
            retrieval=retrieval_method,
            batch_size=args.concurrency,
            rt_writer=rt_writer,
            resume=args.resume,
            disable_retrieval=True,
            force_first_explore=True,
        )
    else:
        results = await eval_search_skill(
            run_items, benchmark, config, model_profile,
            args.concurrency, args.max_rounds,
            skill_library_path, skill_top_k,
            retrieval=retrieval_method,
            batch_size=args.concurrency,
            rt_writer=rt_writer,
            resume=args.resume,
            disable_retrieval=True,
        )

    overall = _agg(results)
    print(f"\n{'='*60}")
    print(f"  Train phase complete: {overall['correct']}/{overall['n']} "
          f"= {_fmt_pct(overall['acc'])}%")
    print(f"  Skills collected in: {skill_library_path}")
    if os.path.exists(skill_library_path):
        with open(skill_library_path, "r") as f:
            skills = json.load(f)
        print(f"  Total skills in library: {len(skills)}")
    print(f"{'='*60}")

    return results


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def print_page_visit_stats(label: str, baseline: Dict = None):
    pv = get_page_visit_stats()
    if baseline:
        pv = {k: pv[k] - baseline.get(k, 0) for k in pv}
    if pv["total"] <= 0:
        return
    effective = pv["success"] + pv["failed"]
    rate = pv["success"] / effective * 100 if effective > 0 else 0.0
    forbid_rate = pv["forbidden"] / pv["total"] * 100
    print(f"  [PageVisit:{label}] total={pv['total']}  "
          f"success={pv['success']}  failed={pv['failed']}  "
          f"forbidden={pv['forbidden']}  "
          f"success_rate={rate:.1f}%  forbidden_rate={forbid_rate:.1f}%")


def compute_summary(results, mode, args):
    overall = _agg(results)
    by_domain = {k: _agg(v) for k, v in _group_by(results, "domain").items()}
    by_difficulty = {k: _agg(v) for k, v in _group_by(results, "difficulty").items()}
    return {
        "mode": mode,
        "model": args.model,
        "overall": overall,
        "by_discipline": by_domain,
        "by_difficulty": by_difficulty,
    }


def print_mode_summary(summary, mode):
    o = summary["overall"]
    w = 60
    if not o or "correct" not in o:
        print(f"\n{'-'*w}")
        print(f"  Mode: {mode}  |  No results collected (all failed)")
        print(f"{'-'*w}")
        return
    print(f"\n{'-'*w}")
    print(f"  Mode: {mode}  |  {o['correct']}/{o['n']} = {_fmt_pct(o['acc'])}%")
    print(f"  Avg tokens: {o['avg_tokens']:.1f}  |  Avg rounds: {o['avg_rounds']:.2f}")
    if o.get("retrieved_skill_trigger") and o["retrieved_skill_trigger"] != "N/A":
        rate = o.get("retrieved_skill_trigger_rate", 0) or 0
        print(f"  Skill trigger: {o['retrieved_skill_trigger']} = {_fmt_pct(rate)}%")
    if o.get("explore_rate") is not None:
        print(f"  Explore rate: {o['explore_count']}/{o['n']} = {_fmt_pct(o['explore_rate'])}%")
    print(f"{'-'*w}")

    by_d = summary.get("by_difficulty", {})
    if by_d and len(by_d) > 1:
        headers = ["Difficulty", "N", "Correct", "Acc%"]
        rows = []
        for d in sorted(by_d.keys()):
            s = by_d[d]
            rows.append([d[:30], str(s["n"]), str(s["correct"]), _fmt_pct(s["acc"])])
        print(_ascii_table(headers, rows))

    by_domain = summary.get("by_discipline", {})
    if by_domain and len(by_domain) > 1:
        headers = ["Domain", "N", "Correct", "Acc%"]
        rows = []
        for d in sorted(by_domain.keys()):
            s = by_domain[d]
            rows.append([d[:30], str(s["n"]), str(s["correct"]), _fmt_pct(s["acc"])])
        print(_ascii_table(headers, rows))


def print_comparison(all_summaries):
    w = 60
    print(f"\n{'='*w}")
    print(f"  COMPARISON ACROSS MODES")
    print(f"{'='*w}")
    headers = ["Mode", "N", "Correct", "Acc%", "AvgTokens", "AvgRounds"]
    rows = []
    for mode, summary in all_summaries.items():
        o = summary["overall"]
        rows.append([mode, str(o["n"]), str(o["correct"]),
                     _fmt_pct(o["acc"]),
                     f"{o['avg_tokens']:.0f}",
                     f"{o['avg_rounds']:.1f}"])
    print(_ascii_table(headers, rows))
    print(f"{'='*w}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    args = parse_args()
    os.environ["EXSKILL_CONFIG_PATH"] = os.path.abspath(args.config)
    config = load_config(args.config)

    from core.agent import set_quiet
    set_quiet(True)

    model_profile = args.model or config["llm"].get("default")
    args.model = model_profile

    if args.judge_profile:
        os.environ["EXSKILL_REWARD_JUDGE_PRIMARY_PROFILE"] = args.judge_profile

    benchmark = load_benchmark(args.benchmark)

    # Resolve output dir
    model_tag = (model_profile or "default").replace("/", "_")
    profile_cfg = config.get("llm", {}).get("profiles", {}).get(model_profile, {})
    _model_path = profile_cfg.get("model_path", "")
    if _model_path:
        _mp_match = re.search(r'_(\d+_step\d+)', os.path.basename(_model_path))
        if _mp_match:
            model_tag = f"{model_tag}_{_mp_match.group(1)}"

    output_base = args.output or "results/pipeline_eval"

    output_dir = os.path.join(output_base, args.benchmark, model_tag)
    os.makedirs(output_dir, exist_ok=True)

    # ── Main agent mode ──
    skill_cfg = config.get("skill", {})
    skill_top_k = args.skill_top_k or skill_cfg.get("skill_top_k", 3)
    skill_library_path = args.skill_library or os.path.join(output_dir, "skill_library.json")
    if args.skip_train and args.skill_library:
        test_skill_library_path = skill_library_path
    else:
        test_skill_library_path = skill_library_path.replace(".json", "_all.json")

    config["llm"]["default"] = model_profile

    t_start = time.time()

    # =====================================================================
    # Phase 1: Train – skill collection
    # =====================================================================
    if not args.skip_train:
        print(f"\n{'='*60}")
        print(f"  PHASE 1: Skill Collection on Train Set")
        print(f"{'='*60}")

        train_items = load_train_items(benchmark, args)
        print(f"  Loaded {len(train_items)} training questions")
        print(f"  Skill library (train): {skill_library_path}")
        print(f"  Mode: {args.train_mode}")

        train_output_dir = os.path.join(output_dir, "train")
        os.makedirs(train_output_dir, exist_ok=True)

        _pv_baseline = get_page_visit_stats()
        await run_train_phase(
            train_items, benchmark, config, model_profile, args,
            skill_library_path, train_output_dir,
        )
        print_page_visit_stats("train", baseline=_pv_baseline)
    else:
        print(f"\n[Skip] Training phase skipped. Using skill library: {test_skill_library_path}")
        if not os.path.exists(test_skill_library_path):
            print(f"  [Warning] Skill library not found at {test_skill_library_path}")

    # =====================================================================
    # Phase 2: Test – evaluate with multiple modes
    # =====================================================================
    if args.skip_test:
        print(f"\n[Skip] Test phase skipped (--skip_test).")
        print(f"\n{'='*60}")
        print(f"  Pipeline complete (train only)")
        print(f"  Total wall time: {time.time() - t_start:.1f}s")
        print(f"  Output: {output_dir}")
        print(f"  Skill library: {test_skill_library_path}")
        print_page_visit_stats("total")
        print(f"{'='*60}")
        return

    print(f"\n{'='*60}")
    print(f"  PHASE 2: Evaluation on Test Set")
    print(f"{'='*60}")

    test_items = load_test_items(benchmark, args)
    print(f"  Loaded {len(test_items)} test questions")
    print(f"  Test modes: {args.test_mode}")

    skill_modes = {"skill_only", "skill_only_no_code", "search_skill", "search_skill_no_code"}
    has_skill_test = bool(set(args.test_mode) & skill_modes) and os.path.exists(test_skill_library_path)
    if has_skill_test:
        import hashlib
        lib_abs = os.path.abspath(test_skill_library_path)
        lib_hash = hashlib.md5(lib_abs.encode()).hexdigest()[:4]
        test_output_dir = os.path.join(output_dir, f"test-{lib_hash}")
    else:
        test_output_dir = os.path.join(output_dir, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    if has_skill_test:
        run_config = {
            "skill_library": os.path.abspath(test_skill_library_path),
            "test_modes": args.test_mode,
            "model": model_profile,
            "max_rounds": args.max_rounds,
            "concurrency": args.concurrency,
            "skill_top_k": skill_top_k,
            "skip_train": args.skip_train,
        }
        config_path = os.path.join(test_output_dir, "run_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)
        print(f"  Test output: {test_output_dir}")
        print(f"  Run config:  {config_path}")

    all_summaries = {}
    num_runs = 3 if args.repeat_avg else 1
    repeat_run_summaries = {}

    run_schedule = []
    for m in args.test_mode:
        for ri in range(num_runs):
            run_schedule.append((m, ri))

    for mode_name, run_i in run_schedule:
        run_tag = f"{mode_name}_run{run_i+1}" if num_runs > 1 else mode_name

        print(f"\n{'='*60}")
        if num_runs > 1:
            print(f"  Test mode: {mode_name}  (run {run_i+1}/{num_runs})")
        else:
            print(f"  Test mode: {mode_name}")
        print(f"{'='*60}")

        if num_runs > 1:
            rt_writer = RealtimeWriter(test_output_dir, run_tag)
            rt_writer.prepare(resume=False)
            mode_items = test_items
        else:
            rt_writer = RealtimeWriter(test_output_dir, mode_name)
            rt_writer.prepare(resume=args.resume)

            if args.resume:
                completed_ids = rt_writer.get_completed_ids()
                mode_items = [it for it in test_items if it["id"] not in completed_ids]
                if completed_ids:
                    print(f"  [Resume] {len(completed_ids)} completed, "
                          f"{len(mode_items)} remaining")
                if not mode_items:
                    print(f"  [Resume] All done for '{mode_name}', skipping.")
                    all_results = _load_mode_results(rt_writer, mode_name)
                    if all_results:
                        summary = compute_summary(all_results, mode_name, args)
                        print_mode_summary(summary, mode_name)
                        all_summaries[mode_name] = summary
                    continue
            else:
                mode_items = test_items

        _pv_baseline_mode = get_page_visit_stats()

        if mode_name == "direct":
            llm = LLMClient.from_config(config, profile_name=model_profile)
            attach_usage_tracking(llm)
            results = await eval_direct(mode_items, benchmark, llm,
                                        args.concurrency, rt_writer=rt_writer)

        elif mode_name == "code":
            results = await eval_code(mode_items, benchmark, config, model_profile,
                                      args.concurrency, args.max_rounds,
                                      rt_writer=rt_writer)

        elif mode_name == "search_code":
            results = await eval_search_code(mode_items, benchmark, config, model_profile,
                                             args.concurrency, args.max_rounds,
                                             rt_writer=rt_writer)

        elif mode_name == "free_tool":
            results = await eval_free_tool(mode_items, benchmark, config, model_profile,
                                           args.concurrency, args.max_rounds,
                                           rt_writer=rt_writer)

        elif mode_name == "search_skill":
            test_skill_lib = os.path.join(test_output_dir, "skill_library_search_skill.json")
            if not args.resume or not os.path.exists(test_skill_lib):
                if os.path.exists(test_skill_library_path):
                    shutil.copy2(test_skill_library_path, test_skill_lib)
                    print(f"  [search_skill] Copied train skill library -> {test_skill_lib}")
                else:
                    test_skill_lib = test_skill_library_path
            results = await eval_search_skill(
                mode_items, benchmark, config, model_profile,
                args.concurrency, args.max_rounds,
                test_skill_lib, skill_top_k,
                retrieval=skill_cfg.get("retrieval", "embedding"),
                batch_size=args.concurrency,
                rt_writer=rt_writer,
                resume=True,
            )

        elif mode_name == "search_skill_no_code":
            test_skill_lib = os.path.join(test_output_dir, "skill_library_search_skill_no_code.json")
            if not args.resume or not os.path.exists(test_skill_lib):
                if os.path.exists(test_skill_library_path):
                    shutil.copy2(test_skill_library_path, test_skill_lib)
                    print(f"  [search_skill_no_code] Copied train skill library -> {test_skill_lib}")
                else:
                    test_skill_lib = test_skill_library_path
            results = await eval_search_skill_no_code(
                mode_items, benchmark, config, model_profile,
                args.concurrency, args.max_rounds,
                test_skill_lib, skill_top_k,
                retrieval=skill_cfg.get("retrieval", "embedding"),
                batch_size=args.concurrency,
                rt_writer=rt_writer,
                resume=True,
            )

        elif mode_name == "skill_only":
            results = await eval_skill_only(
                mode_items, benchmark, config, model_profile,
                args.concurrency, args.max_rounds,
                test_skill_library_path, skill_top_k,
                rt_writer=rt_writer,
            )

        elif mode_name == "skill_only_no_code":
            results = await eval_skill_only_no_code(
                mode_items, benchmark, config, model_profile,
                args.concurrency, args.max_rounds,
                test_skill_library_path, skill_top_k,
                rt_writer=rt_writer,
            )

        # Compute summary
        if args.resume and num_runs == 1:
            all_results = _load_mode_results(rt_writer, mode_name)
        else:
            all_results = results

        summary = compute_summary(all_results, run_tag, args)
        print_page_visit_stats(f"test:{run_tag}", baseline=_pv_baseline_mode)

        if num_runs > 1:
            o = summary["overall"]
            print(f"  Run {run_i+1}/{num_runs}: "
                  f"{o['correct']}/{o['n']} = {_fmt_pct(o['acc'])}%")
            repeat_run_summaries.setdefault(mode_name, []).append(summary)
        else:
            print_mode_summary(summary, mode_name)
            all_summaries[mode_name] = summary

    # Print repeat average summary
    if args.repeat_avg and repeat_run_summaries:
        print(f"\n{'='*60}")
        print(f"  REPEAT AVERAGE SUMMARY ({num_runs} runs)")
        print(f"{'='*60}")
        avg_summary_lines = []
        for mn, summaries in repeat_run_summaries.items():
            accs = [s["overall"]["acc"] for s in summaries]
            avg_acc = sum(accs) / len(accs)
            std_acc = (sum((a - avg_acc) ** 2 for a in accs) / len(accs)) ** 0.5
            runs_str = ", ".join(f"{a*100:.1f}%" for a in accs)
            line = (f"  {mn}: [{runs_str}]  "
                    f"avg={avg_acc*100:.1f}% ± {std_acc*100:.1f}%")
            print(line)
            avg_summary_lines.append(line)
        print(f"{'='*60}")

        avg_txt_path = os.path.join(test_output_dir, "repeat_avg_summary.txt")
        with open(avg_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Repeat average summary ({num_runs} runs)\n")
            f.write(f"{'='*60}\n")
            for line in avg_summary_lines:
                f.write(line + "\n")
        print(f"  [Saved] {avg_txt_path}")

    # Final comparison (single-run mode only)
    if num_runs == 1 and len(all_summaries) > 1:
        print_comparison(all_summaries)

    # Save comparison JSON
    comparison_path = os.path.join(test_output_dir, "comparison.json")
    comparison_data = {}
    for mode_name, summary in all_summaries.items():
        o = summary["overall"]
        comparison_data[mode_name] = {
            "n": o["n"],
            "correct": o["correct"],
            "accuracy": o["acc"],
            "avg_tokens": o["avg_tokens"],
            "avg_rounds": o["avg_rounds"],
        }
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    print(f"\n  [Saved] {comparison_path}")

    # Save human-readable summary txt
    summary_path = os.path.join(test_output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Benchmark: {args.benchmark}  Model: {args.model}\n")
        f.write(f"{'='*60}\n")
        for mode_name, summary in all_summaries.items():
            o = summary["overall"]
            f.write(f"\n  Mode: {mode_name}  |  {o['correct']}/{o['n']} = {_fmt_pct(o['acc'])}%\n")
            f.write(f"  Avg tokens: {o['avg_tokens']:.1f}  |  Avg rounds: {o['avg_rounds']:.2f}\n")
            if o.get("retrieved_skill_trigger") and o["retrieved_skill_trigger"] != "N/A":
                rate = o.get("retrieved_skill_trigger_rate", 0) or 0
                f.write(f"  Skill trigger: {o['retrieved_skill_trigger']} = {_fmt_pct(rate)}%\n")
            if o.get("explore_rate") is not None:
                f.write(f"  Explore rate: {o['explore_count']}/{o['n']} = {_fmt_pct(o['explore_rate'])}%\n")
            by_d = summary.get("by_difficulty", {})
            if by_d and len(by_d) > 1:
                f.write(f"\n  {'Difficulty':<30} {'N':>5} {'Correct':>8} {'Acc%':>6}\n")
                for d in sorted(by_d.keys()):
                    s = by_d[d]
                    f.write(f"  {d[:30]:<30} {s['n']:>5} {s['correct']:>8} {_fmt_pct(s['acc']):>6}%\n")
            by_domain = summary.get("by_discipline", {})
            if by_domain and len(by_domain) > 1:
                f.write(f"\n  {'Domain':<30} {'N':>5} {'Correct':>8} {'Acc%':>6}\n")
                for d in sorted(by_domain.keys()):
                    s = by_domain[d]
                    f.write(f"  {d[:30]:<30} {s['n']:>5} {s['correct']:>8} {_fmt_pct(s['acc']):>6}%\n")
            f.write(f"{'─'*60}\n")
    print(f"  [Saved] {summary_path}")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete")
    print(f"  Total wall time: {time.time() - t_start:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"  Skill library (test): {test_skill_library_path}")
    print_page_visit_stats("total")
    print(f"{'='*60}")


def _load_mode_results(rt_writer: RealtimeWriter, mode_name: str) -> List[Dict]:
    """Load all results for a mode from the JSONL file."""
    results = []
    if os.path.exists(rt_writer.jsonl_path):
        with open(rt_writer.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("mode") == mode_name:
                        results.append(obj)
                except json.JSONDecodeError:
                    continue
    return results


if __name__ == "__main__":
    asyncio.run(main())
