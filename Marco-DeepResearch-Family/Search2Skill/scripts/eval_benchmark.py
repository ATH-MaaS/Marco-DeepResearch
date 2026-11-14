"""Unified benchmark evaluation script with multi-mode support.

Modes (tool ablation):
  direct              -- Pure LLM zero-shot (no agent, no tools).
  code                -- Agent + Code Execution only (force_internal, no web search).
  search_skill        -- Agent + Code + Web Search + Skill Generation + Retrieval
                         (persistent skill library, retrieval via --retrieval fuzzy|embedding).
  full                -- Meta-mode: run direct + code + free_tool + search_skill.

Metrics tracked per question:
  - correct (bool), score (float)
  - prompt_tokens, completion_tokens, total_tokens
  - web_search_count
  - num_rounds

Aggregated summary includes:
  - accuracy, avg_tokens_per_question, avg_web_searches, avg_rounds, per-domain breakdown

Usage:
    # Direct LLM baseline
    python scripts/eval_benchmark.py --benchmark supergpqa --mode direct --model qwen3-8b --num 200

    # Code-only (agent + code execution)
    python scripts/eval_benchmark.py --benchmark supergpqa --mode code --model qwen3-8b --num 200

    # Web search + code + skill gen (retrieval method from config.yaml)
    python scripts/eval_benchmark.py --benchmark supergpqa --mode search_skill --model qwen3-8b --num 200

    # Run ALL four modes sequentially
    python scripts/eval_benchmark.py --benchmark supergpqa --mode full --model qwen3-8b --num 200

    # Filter by discipline
    python scripts/eval_benchmark.py --benchmark supergpqa --mode direct --num 100 --subset Physics

    # List available subsets
    python scripts/eval_benchmark.py --benchmark supergpqa --list_subsets
"""

import argparse
import asyncio
import functools
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import yaml

# Force unbuffered stdout for nohup / redirect
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Suppress noisy library logs — only show CRITICAL
logging.basicConfig(level=logging.CRITICAL, force=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.client import LLMClient
from benchmarks import load_benchmark
from core.decision import extract_answer_from_tag, FINAL_ANSWER_PROMPT
from evaluation.usage_tracker import UsageTracker, attach_usage_tracking, set_current_tracker


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

async def _async_evaluate(benchmark, prediction: str, ground_truth: str,
                          **kwargs) -> Dict:
    """Run benchmark.evaluate() in a thread so sync LLM judges (e.g. OmniMath's
    cascade_answer_judge) don't block the async event loop."""
    loop = asyncio.get_running_loop()
    fn = functools.partial(benchmark.evaluate, prediction, ground_truth, **kwargs)
    return await loop.run_in_executor(None, fn)


DIRECT_ANSWER_PROMPT = """\
Please answer the following question. Think step by step, then put your final answer inside <ANSWER></ANSWER> tags.

For multiple-choice questions, put only the option letter inside the tags, e.g. <ANSWER>B</ANSWER>.
For open-ended questions, put a concise answer inside the tags.

{question}"""


# ---------------------------------------------------------------------------
# Real-time result writer – appends each result to disk as it completes
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "mode", "id", "domain", "difficulty", "subfield", "correct", "score",
    "total_tokens", "prompt_tokens", "completion_tokens",
    "web_search_count", "num_rounds",
    "explore_triggered", "num_queries", "num_page_visits",
    "skills_generated", "skills_retrieved", "retrieved_skill_used",
    "teacher_rewrite_count",
    "ground_truth", "prediction",
]


class RealtimeWriter:
    """Appends results to JSONL and CSV as they complete, so partial
    progress is preserved even if the process is interrupted.

    Usage::

        writer = RealtimeWriter(output_dir, mode)
        writer.prepare()          # remove old mode data, write CSV header
        ...                       # during eval:
        writer.write_one(result)  # append each result immediately
        # At the end, summary.xlsx is written by save_results().
    """

    def __init__(self, output_dir: str, mode: str):
        self.output_dir = output_dir
        self.mode = mode
        self.jsonl_path = os.path.join(output_dir, "results.jsonl")
        self.csv_path = os.path.join(output_dir, "detail.csv")

    # ---- called once before eval starts ----

    def prepare(self, resume: bool = False):
        """Remove existing results for this mode from output files.
        
        If resume=True, skip cleanup and preserve existing results so that
        already-completed questions can be skipped.
        """
        import csv
        os.makedirs(self.output_dir, exist_ok=True)

        if resume:
            # In resume mode, don't clean anything — just ensure CSV header exists
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
                    writer.writeheader()
            return

        # Clean JSONL: remove old mode entries, keep others
        existing_lines: list[str] = []
        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("mode") != self.mode:
                            existing_lines.append(line)
                    except json.JSONDecodeError:
                        existing_lines.append(line)
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            for line in existing_lines:
                f.write(line + "\n")

        # Clean CSV: remove old mode rows, keep others
        existing_rows: list[dict] = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("mode") != self.mode:
                        existing_rows.append(row)
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)

    def get_completed_ids(self) -> set:
        """Read existing JSONL and return set of question IDs already completed for this mode."""
        completed = set()
        if not os.path.exists(self.jsonl_path):
            return completed
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("mode") == self.mode and "id" in obj:
                        completed.add(obj["id"])
                except json.JSONDecodeError:
                    continue
        return completed

    # ---- called for each completed result ----

    def write_one(self, result: Dict):
        """Append a single result to JSONL and CSV immediately."""
        import csv as _csv
        result["mode"] = self.mode

        # JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # CSV
        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writerow(result)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _resolve_env_vars(config)
    return config


def _resolve_env_vars(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                obj[k] = os.environ.get(v[2:-1], v)
            else:
                _resolve_env_vars(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                obj[i] = os.environ.get(v[2:-1], v)
            else:
                _resolve_env_vars(v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Unified benchmark evaluation (multi-mode)")

    # Required
    p.add_argument("--benchmark", type=str, required=True,
                   help="Benchmark name (supergpqa, mmlu_pro, finance_bench, ...)")
    ALL_MODES = ["direct", "code", "free_tool", "search_skill", "search_skill_no_code", "skill_only", "skill_only_no_code"]
    p.add_argument("--mode", type=str, nargs="+", default=["direct", "code", "search_skill"],
                   choices=ALL_MODES + ["full"],
                   help="Evaluation mode(s). 'full' expands to all modes. "
                        "Specify multiple: --mode direct code search_skill")
    # Model & data
    p.add_argument("--model", type=str, default=None,
                   help="LLM profile name (default: config default)")
    p.add_argument("--num", type=int, default=-1,
                   help="Number of questions to evaluate (-1 = all)")
    p.add_argument("--subset", type=str, default=None,
                   help="Filter by subset/discipline")
    p.add_argument("--data_file", type=str, default=None,
                   help="Load pre-extracted JSON data file instead of HuggingFace "
                        "(e.g. data/supergpqa_test_1500.json)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle_seed", type=int, default=None,
                   help="If set, shuffle loaded questions with this seed before evaluation")
    # Agent settings
    p.add_argument("--max_rounds", type=int, default=15,
                   help="Max agent rounds per question (code/search_skill)")
    p.add_argument("--concurrency", type=int, default=20,
                   help="Max concurrent evaluations")
    # Skill mode settings
    p.add_argument("--skill_library", type=str, default=None,
                   help="Persistent skill library path for search_skill mode (default: auto)")
    p.add_argument("--skill_top_k", type=int, default=None,
                   help="Number of skills to retrieve per question (default: from config)")
    p.add_argument("--skill_batch_size", type=int, default=20,
                   help="Wave batch size for search_skill mode. "
                        "Questions within a wave run concurrently; skills accumulate across waves. "
                        "(default: 20, set 1 for fully sequential)")
    # IO
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--output", type=str, default=None,
                   help="Output base directory (default: results/eval)")
    p.add_argument("--list_subsets", action="store_true",
                   help="List available subsets and exit")
    p.add_argument("--resume", action="store_true",
                   help="Resume from previous run: skip already-completed question IDs "
                        "found in the output JSONL file instead of clearing old results")
    # Judge settings (for open-ended benchmarks like OmniMath)
    p.add_argument("--judge_profile", type=str, default=None,
                   help="LLM profile for cascade answer judge (default: from config rl section)")
    p.add_argument("--repeat_avg", action="store_true",
                   help="Run each mode 3 times and report average accuracy (reduces variance on small test sets)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Mode 1: Direct -- pure LLM zero-shot
# ---------------------------------------------------------------------------

async def eval_direct(items, benchmark, llm: LLMClient,
                      concurrency: int,
                      rt_writer: Optional[RealtimeWriter] = None) -> List[Dict]:
    """Pure LLM zero-shot evaluation."""
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def _one(item):
        question_text = benchmark.format_question(item)
        ground_truth = benchmark.get_ground_truth(item)

        tracker = UsageTracker()
        set_current_tracker(tracker)
        async with sem:
            t0 = time.time()
            messages = [{"role": "user",
                         "content": DIRECT_ANSWER_PROMPT.format(question=question_text)}]
            response = await llm.chat(messages, temperature=0.3)
            response = response or ""
            messages.append({"role": "assistant", "content": response})

            # Extract answer from <ANSWER> tag; if missing, ask once more
            answer = extract_answer_from_tag(response)
            if not answer:
                messages.append({"role": "user", "content": FINAL_ANSWER_PROMPT})
                followup = await llm.chat(messages, temperature=0.3)
                followup = followup or ""
                messages.append({"role": "assistant", "content": followup})
                answer = extract_answer_from_tag(followup)
                if not answer:
                    # Last resort: use raw response stripped of tags
                    answer = response.strip()

            elapsed = time.time() - t0
        usage = tracker.snapshot()

        judge_t0 = time.time()
        eval_result = await _async_evaluate(benchmark, answer, ground_truth,
                                             question=question_text)
        judge_elapsed = time.time() - judge_t0
        meta = item.get("metadata", {})
        return {
            "id": item["id"],
            "domain": item.get("domain", ""),
            "difficulty": meta.get("difficulty", ""),
            "subfield": meta.get("subfield", ""),
            "question": item["question"][:200],
            "ground_truth": ground_truth,
            "prediction": answer[:500],
            "correct": eval_result["correct"],
            "score": eval_result["score"],
            "detail": eval_result["detail"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "num_llm_calls": usage["num_calls"],
            "web_search_count": 0,
            "num_rounds": 1,
            "conversation": messages,
        }

    tasks = [asyncio.create_task(_one(item)) for item in items]
    done_count = 0
    correct_count = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        if rt_writer:
            rt_writer.write_one(result)
        done_count += 1
        if result["correct"]:
            correct_count += 1
        acc = correct_count / done_count * 100
        sym = "+" if result["correct"] else "-"
        print(f"  [direct] [{done_count}/{len(items)}] {sym}  "
              f"acc={acc:.1f}%  tokens={result['total_tokens']}")

    return results


# ---------------------------------------------------------------------------
# Mode 2: Code -- agent + code execution only
# ---------------------------------------------------------------------------

async def eval_code(items, benchmark, config: dict, model_profile: str,
                    concurrency: int, max_rounds: int,
                    rt_writer: Optional[RealtimeWriter] = None) -> List[Dict]:
    """Agent with ONLY Code Execution tool."""
    from core.agent import ExplorationAgent
    from core.skill_manager import SkillManager
    from tools.code_executor import CodeExecutor

    class _NoopWebSearch:
        async def search(self, query: str):
            return []
        async def batch_search(self, querys):
            return {}

    class _NoopWebVisit:
        async def visit(self, urls, goal: str = ""):
            return []

    student_llm = LLMClient.from_config(config, profile_name=model_profile)
    attach_usage_tracking(student_llm)
    code_executor = CodeExecutor.from_config(config)
    skill_cfg = config.get("skill", {})

    def _make_agent():
        temp_lib = os.path.join(tempfile.gettempdir(), f"eval_bm_code_{id(object())}.json")
        sm = SkillManager(
            library_path=temp_lib,
            max_skills=skill_cfg.get("max_skills", 200),
            similarity_threshold=skill_cfg.get("similarity_threshold", 0.7),
        )
        sm.skills = []
        return ExplorationAgent(
            llm=student_llm,
            skill_manager=sm,
            web_search=_NoopWebSearch(),
            web_visit=_NoopWebVisit(),
            code_executor=code_executor,
            max_rounds=max_rounds,
            explore_max_searches=0,
            force_internal=True,
            fetch_top_k=1,
        )

    sem = asyncio.Semaphore(concurrency)
    results = []

    async def _one(item):
        question_text = benchmark.format_question(item)
        ground_truth = benchmark.get_ground_truth(item)

        tracker = UsageTracker()
        set_current_tracker(tracker)
        async with sem:
            agent = _make_agent()
            t0 = time.time()
            try:
                agent_result = await agent.solve(
                    question_text,
                    override_skills=[],
                )
                answer = agent_result.get("answer", "")
                num_rounds = agent_result.get("num_rounds", 0)
                conversation = agent_result.get("conversation_history", [])
            except Exception as e:
                print(f"  [code] [Error] {item['id']}: {e}")
                answer, num_rounds, conversation = "", 0, []
            elapsed = time.time() - t0
        usage = tracker.snapshot()

        judge_t0 = time.time()
        eval_result = await _async_evaluate(benchmark, answer, ground_truth,
                                             question=question_text)
        judge_elapsed = time.time() - judge_t0

        # Cleanup temp library
        lib_path = agent.skill_manager.library_path
        if os.path.exists(lib_path):
            try:
                os.remove(lib_path)
            except OSError:
                pass

        return {
            "id": item["id"],
            "domain": item.get("domain", ""),
            "difficulty": item.get("metadata", {}).get("difficulty", ""),
            "subfield": item.get("metadata", {}).get("subfield", ""),
            "question": item["question"][:200],
            "ground_truth": ground_truth,
            "prediction": answer[:500],
            "correct": eval_result["correct"],
            "score": eval_result["score"],
            "detail": eval_result["detail"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "num_llm_calls": usage["num_calls"],
            "web_search_count": 0,
            "num_rounds": num_rounds,
            "conversation": conversation,
        }

    tasks = [asyncio.create_task(_one(item)) for item in items]
    done_count = 0
    correct_count = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        if rt_writer:
            rt_writer.write_one(result)
        done_count += 1
        if result["correct"]:
            correct_count += 1
        acc = correct_count / done_count * 100
        sym = "+" if result["correct"] else "-"
        print(f"  [code] [{done_count}/{len(items)}] {sym}  "
              f"acc={acc:.1f}%  rounds={result['num_rounds']}  tokens={result['total_tokens']}")

    return results


# ---------------------------------------------------------------------------
# LLM-based skill relevance filter
# ---------------------------------------------------------------------------

def _build_filter_llm(config: dict, filter_profile: str):
    """Build a filter LLM client (or list of clients for multi-key load balancing).

    `filter_profile` may be a single profile name (e.g. "qwen37-max") or a
    comma-separated list (e.g. "qwen37-max,qwen37-max2"). The latter returns
    a list of LLMClient instances that `_llm_filter_skills` will round-robin
    across to spread QPS over multiple API keys / endpoints.
    """
    profiles = [p.strip() for p in str(filter_profile or "").split(",") if p.strip()]
    if not profiles:
        return None
    clients = []
    for p in profiles:
        client = LLMClient.from_config(config, profile_name=p)
        attach_usage_tracking(client)
        clients.append(client)
    print(f"[FilterLLM] Built {len(clients)} client(s): {profiles}")
    return clients[0] if len(clients) == 1 else clients


def _get_filter_llm(config: dict, fallback_profile: str):
    """Build filter LLM from config with unified resolution path."""
    skill_cfg = config.get("skill", {})
    wv_cfg = config.get("web_visit", {})
    filter_profile = (skill_cfg.get("filter_llm")
                      or wv_cfg.get("compress_llm")
                      or fallback_profile)
    return _build_filter_llm(config, filter_profile)


async def _llm_filter_skills(llm, question: str,
                              skills: List[Dict]) -> Tuple[List[Dict], str]:
    """Use LLM to judge which embedding-retrieved skills are truly relevant.

    `llm` accepts either a single LLMClient or a list of LLMClients. When a list
    is given, a random client is chosen per call to spread QPS across multiple
    API keys/endpoints.

    Returns (kept_skills, reason). reason is the LLM's explanation string
    (may be empty if the LLM returned nothing parseable).
    """
    if not skills:
        return [], ""

    # Pick one client when a list is passed (multi-key load balancing)
    if isinstance(llm, (list, tuple)):
        import random as _random
        llm = _random.choice(list(llm))

    from core.skill_schema import skill_use_when, skill_workflow, skill_code

    skill_blocks = []
    for i, s in enumerate(skills):
        name = s.get("skill_name", "")
        use_when = skill_use_when(s)
        workflow = skill_workflow(s)
        code = skill_code(s).strip()
        source_q = s.get("source_question", "") or ""
        code_sig = "\n".join(code.split("\n")[:8]) if code else ""

        block = f"Skill {i}:\n  name: {name}\n  use_when: {use_when}"
        if source_q:
            block += f"\n  source_question: {source_q}"
        if workflow:
            block += f"\n  workflow: {workflow}"
        if code_sig:
            block += f"\n  code_signature: {code_sig}"
        skill_blocks.append(block)

    MAX_KEEP = 2

    prompt = (
        "You are an expert skill relevance judge. For EACH candidate skill below, "
        "decide whether applying it would HELP, be NEUTRAL, or actively MISLEAD the "
        "solver on the current question. You MUST output a verdict for every "
        "candidate (no skipping).\n\n"

        f"HARD LIMIT: at most {MAX_KEEP} skills may end up labeled HELPFUL. "
        "When nothing is a precise fit, mark everything NEUTRAL/HARMFUL.\n\n"

        "## Labels (assign exactly one per skill)\n"
        "- HELPFUL: The skill's specific method/algorithm/procedure DIRECTLY solves "
        "the current question (same problem structure and constraints, not just same topic).\n"
        "- NEUTRAL: Topic overlap but the solution method is different or "
        "inapplicable — will not help.\n"
        "- HARMFUL: The skill's method would ACTIVELY MISLEAD the solver "
        "(wrong framework, wrong assumptions that conflict with the question's "
        "constraints, e.g. skill assumes rotational symmetry but the question has a "
        "fixed starting point; skill solves monotone paths but the question needs "
        "Hamiltonian paths; skill's code API doesn't match the state space).\n\n"

        "## Key decision rule\n"
        "Compare each skill's source_question with the current question:\n"
        "- If the SOLVING APPROACH (not just topic) is the same → HELPFUL\n"
        "- If only the topic matches but a different method is needed → NEUTRAL\n"
        "- If using the skill's method would lead to a wrong answer → HARMFUL\n\n"

        "Only HELPFUL skills will be kept. Be conservative; favor NEUTRAL over HELPFUL "
        "when in doubt.\n\n"

        f"## Current Question:\n{question}\n\n"
        "## Candidate Skills:\n\n"
        + "\n\n---\n\n".join(skill_blocks) + "\n\n"
        '## Output\n'
        'Output STRICT JSON only — no prose, no markdown fences. Provide one entry '
        'per candidate, in order:\n'
        '{"verdicts": [{"idx": <int>, "label": "HELPFUL|NEUTRAL|HARMFUL", '
        '"reason": "<one short clause>"}, ...]}\n'
        f'Use indices 0..{len(skills)-1}. Every index must appear exactly once. '
        f'At most {MAX_KEEP} entries may be labeled HELPFUL.\n'
    )

    response = await llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=4096,
    )

    if not response:
        print(f"[SkillFilter] LLM returned empty, filtering all {len(skills)}")
        return [], ""

    # Strip <think>...</think> tags (thinking mode generates these)
    answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    # Handle truncated thinking (no closing tag)
    if "<think>" in answer:
        answer = ""
    # Strip ```json ... ``` fences if present
    answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip(), flags=re.MULTILINE)

    # Brace-balanced extraction of the first top-level JSON object.
    def _extract_json_obj(text: str) -> str:
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start:i + 1]
        return ""

    # Parse JSON response — new schema: {"verdicts": [{idx, label, reason}, ...]}
    # Legacy schema {"skills": [<idx>...], "reason": "..."} is still accepted.
    import json as _json
    relevant_indices = set()
    reason = ""
    obj_text = _extract_json_obj(answer)
    parsed = None
    if obj_text:
        try:
            parsed = _json.loads(obj_text)
        except (_json.JSONDecodeError, TypeError, ValueError):
            parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("verdicts"), list):
        reason_parts = []
        for v in parsed["verdicts"]:
            if not isinstance(v, dict):
                continue
            raw_idx = v.get("idx")
            if isinstance(raw_idx, int):
                k = raw_idx
            elif isinstance(raw_idx, str) and raw_idx.strip().isdigit():
                k = int(raw_idx.strip())
            else:
                continue
            if not (0 <= k < len(skills)):
                continue
            label = str(v.get("label", "")).strip().upper()
            v_reason = str(v.get("reason", "")).strip()
            reason_parts.append(f"[{k}:{label}] {v_reason}")
            if label == "HELPFUL":
                relevant_indices.add(k)
        reason = "; ".join(reason_parts)
    elif isinstance(parsed, dict):
        # Legacy fallback
        reason = parsed.get("reason", "") or ""
        for idx in parsed.get("skills", []) or []:
            if isinstance(idx, int) and 0 <= idx < len(skills):
                relevant_indices.add(idx)
            elif isinstance(idx, str) and idx.strip().isdigit():
                k = int(idx.strip())
                if 0 <= k < len(skills):
                    relevant_indices.add(k)
    else:
        # Last-resort fallback: any standalone integer becomes an index.
        for token in re.findall(r"\b\d+\b", answer):
            k = int(token)
            if 0 <= k < len(skills):
                relevant_indices.add(k)
        print(f"[SkillFilter] JSON parse failed, fallback indices={sorted(relevant_indices)}. "
              f"Raw[:300]={answer[:300]!r}")

    if not relevant_indices:
        return [], reason

    # Hard cap, even if the LLM ignored MAX_KEEP.
    kept_indices = sorted(relevant_indices)
    if len(kept_indices) > MAX_KEEP:
        print(f"[SkillFilter] LLM returned {len(kept_indices)} skills, "
              f"capping to {MAX_KEEP}. Raw[:200]={answer[:200]!r}")
        kept_indices = kept_indices[:MAX_KEEP]

    kept = [skills[i] for i in kept_indices]
    profile_tag = getattr(llm, "profile", None) or getattr(llm, "profile_name", None) or "?"
    print(f"[SkillFilter:{profile_tag}] LLM kept {len(kept)}/{len(skills)} skills: "
          f"{[s.get('skill_name', '') for s in kept]}. Reason: {reason}")
    return kept, reason


# ---------------------------------------------------------------------------
# Mode: Skill Only -- agent + code + activate_skill (no web search, no skill gen)
# ---------------------------------------------------------------------------

async def eval_skill_only(items, benchmark, config: dict, model_profile: str,
                          concurrency: int, max_rounds: int,
                          skill_library_path: Optional[str],
                          skill_top_k: int,
                          rt_writer: Optional[RealtimeWriter] = None) -> List[Dict]:
    """Agent with code execution + ACTIVATE_SKILL only (frozen skill library, no web search).

    Uses embedding retrieval to find relevant skills per question from the
    pre-built skill library. No web search, no page visit, no skill generation.
    """
    from core.agent import ExplorationAgent
    from core.skill_manager import SkillManager
    from core.skill_schema import normalize_skill
    from tools.code_executor import CodeExecutor

    class _NoopWebSearch:
        async def search(self, query: str):
            return []
        async def batch_search(self, querys):
            return {}

    class _NoopWebVisit:
        async def visit(self, urls, goal: str = ""):
            return []

    student_llm = LLMClient.from_config(config, profile_name=model_profile)
    attach_usage_tracking(student_llm)
    code_executor = CodeExecutor.from_config(config)

    filter_llm = _get_filter_llm(config, model_profile)

    skill_cfg = config.get("skill", {})

    # Load frozen skill library
    frozen_skills = []
    if skill_library_path and os.path.exists(skill_library_path):
        with open(skill_library_path, "r", encoding="utf-8") as f:
            import json as _json
            frozen_skills = _json.load(f)
        frozen_skills = [normalize_skill(s) for s in frozen_skills]
    print(f"[SkillOnly] Loaded {len(frozen_skills)} skills from {skill_library_path}")

    # Embedding retriever
    use_embedding = False
    retriever = None
    retrieval_method = skill_cfg.get("retrieval", "embedding")
    if retrieval_method == "embedding":
        from evaluation.embedding_retriever import EmbeddingRetriever
        retriever = EmbeddingRetriever.from_config(config)
        if retriever and frozen_skills:
            use_embedding = True
            await retriever.index_skills(frozen_skills)
            print(f"[SkillOnly] Embedding retrieval indexed {len(frozen_skills)} skills")

    if not use_embedding:
        frozen_sm = SkillManager(
            library_path=skill_library_path or "",
            max_skills=skill_cfg.get("max_skills", 200),
            similarity_threshold=skill_cfg.get("similarity_threshold", 0.7),
        )
        frozen_sm.skills = list(frozen_skills)

    sem = asyncio.Semaphore(concurrency)
    results = []

    async def _one(item):
        question_text = benchmark.format_question(item)
        ground_truth = benchmark.get_ground_truth(item)

        tracker = UsageTracker()
        set_current_tracker(tracker)
        async with sem:
            # Retrieve skills
            retrieval_score_details = []
            if frozen_skills:
                if use_embedding:
                    retrieved_ws = await retriever.retrieve_with_scores(
                        question_text, top_k=skill_top_k,
                        threshold=skill_cfg.get("similarity_threshold", 0.3))
                    retrieval_score_details, _, _ = _extract_retrieval_scores(retrieved_ws)
                    retrieved = [s for s, _ in retrieved_ws]
                else:
                    retrieved = frozen_sm.retrieve(question_text, top_k=skill_top_k)
            else:
                retrieved = []

            # LLM relevance filter: use stronger model for better judgment
            skill_filter_reason = ""
            if retrieved:
                retrieved, skill_filter_reason = await _llm_filter_skills(
                    filter_llm, question_text, retrieved)

            # Create agent with retrieved skills, force_internal (no explore)
            temp_lib = os.path.join(tempfile.gettempdir(), f"eval_bm_so_{id(object())}.json")
            sm = SkillManager(
                library_path=temp_lib,
                max_skills=skill_cfg.get("max_skills", 200),
                similarity_threshold=skill_cfg.get("similarity_threshold", 0.7),
            )
            sm.skills = [normalize_skill(s) for s in retrieved]

            agent = ExplorationAgent(
                llm=student_llm,
                skill_manager=sm,
                web_search=_NoopWebSearch(),
                web_visit=_NoopWebVisit(),
                code_executor=code_executor,
                max_rounds=max_rounds,
                explore_max_searches=0,
                force_internal=True,
                fetch_top_k=0,
                skip_skill_generation=True,
            )

            t0 = time.time()
            try:
                agent_result = await agent.solve(question_text, context=item.get("context", ""))
                answer = agent_result.get("answer", "")
                num_rounds = agent_result.get("num_rounds", 0)
                conversation = agent_result.get("conversation_history", [])
                called_skill_names = agent_result.get("called_skill_names", [])
                actions = agent_result.get("actions", [])
            except Exception as e:
                print(f"  [skill_only] [Error] {item['id']}: {e}")
                answer, num_rounds, conversation = "", 0, []
                called_skill_names = []
                actions = []
            elapsed = time.time() - t0
        usage = tracker.snapshot()

        # Check if any retrieved skill was actually called and count calls
        retrieved_skill_names_norm = {
            _normalize_skill_name(normalize_skill(s).get("skill_name", ""))
            for s in retrieved
        }
        retrieved_skill_used = bool(
            retrieved and any(n in retrieved_skill_names_norm for n in called_skill_names)
        )
        # Count how many times retrieved skills were called
        skill_call_count = sum(1 for n in called_skill_names if n in retrieved_skill_names_norm)
        called_retrieved_names = [n for n in called_skill_names if n in retrieved_skill_names_norm]

        judge_t0 = time.time()
        eval_result = await _async_evaluate(benchmark, answer, ground_truth,
                                             question=question_text)
        judge_elapsed = time.time() - judge_t0

        if os.path.exists(temp_lib):
            try:
                os.remove(temp_lib)
            except OSError:
                pass

        return {
            "id": item["id"],
            "domain": item.get("domain", ""),
            "difficulty": item.get("metadata", {}).get("difficulty", ""),
            "subfield": item.get("metadata", {}).get("subfield", ""),
            "question": item["question"][:200],
            "ground_truth": ground_truth,
            "prediction": answer[:500],
            "correct": eval_result["correct"],
            "score": eval_result["score"],
            "detail": eval_result["detail"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "num_llm_calls": usage["num_calls"],
            "web_search_count": 0,
            "num_rounds": num_rounds,
            "skills_retrieved": len(retrieved),
            "retrieved_skill_used": retrieved_skill_used,
            "skill_call_count": skill_call_count,
            "called_skill_names": called_retrieved_names,
            "retrieved_skill_scores": retrieval_score_details,
            "filter_kept_skill_names": [
                normalize_skill(s).get("skill_name", "") for s in retrieved
            ],
            "skill_filter_reason": skill_filter_reason,
            "conversation": conversation,
            "elapsed_sec": round(elapsed, 3),
        }

    tasks = [asyncio.create_task(_one(item)) for item in items]
    done_count = 0
    correct_count = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        if rt_writer:
            rt_writer.write_one(result)
        done_count += 1
        if result["correct"]:
            correct_count += 1
        acc = correct_count / done_count * 100
        sym = "+" if result["correct"] else "-"
        retr_tag = "R" if result["retrieved_skill_used"] else "."
        print(f"  [skill_only] [{done_count}/{len(items)}] {sym}{retr_tag}  "
              f"acc={acc:.1f}%  rounds={result['num_rounds']}  "
              f"retrieved={result['skills_retrieved']}  called={result['skill_call_count']}  "
              f"tokens={result['total_tokens']}")

    return results


# ---------------------------------------------------------------------------
# Mode 1b-nc: Skill Only (no code) -- same as skill_only but without code executor
# ---------------------------------------------------------------------------

async def eval_skill_only_no_code(items, benchmark, config: dict, model_profile: str,
                                  concurrency: int, max_rounds: int,
                                  skill_library_path: Optional[str],
                                  skill_top_k: int,
                                  rt_writer: Optional[RealtimeWriter] = None) -> List[Dict]:
    """Agent with ACTIVATE_SKILL only, NO code execution (frozen skill library, no web search)."""
    from core.agent import ExplorationAgent
    from core.skill_manager import SkillManager
    from core.skill_schema import normalize_skill

    class _NoopWebSearch:
        async def search(self, query: str):
            return []
        async def batch_search(self, querys):
            return {}

    class _NoopWebVisit:
        async def visit(self, urls, goal: str = ""):
            return []

    student_llm = LLMClient.from_config(config, profile_name=model_profile)
    attach_usage_tracking(student_llm)

    filter_llm = _get_filter_llm(config, model_profile)

    skill_cfg = config.get("skill", {})

    # Load frozen skill library
    frozen_skills = []
    if skill_library_path and os.path.exists(skill_library_path):
        with open(skill_library_path, "r", encoding="utf-8") as f:
            import json as _json
            frozen_skills = _json.load(f)
        frozen_skills = [normalize_skill(s) for s in frozen_skills]
    print(f"[SkillOnlyNoCode] Loaded {len(frozen_skills)} skills from {skill_library_path}")

    # Embedding retriever
    use_embedding = False
    retriever = None
    retrieval_method = skill_cfg.get("retrieval", "embedding")
    if retrieval_method == "embedding":
        from evaluation.embedding_retriever import EmbeddingRetriever
        retriever = EmbeddingRetriever.from_config(config)
        if retriever and frozen_skills:
            use_embedding = True
            await retriever.index_skills(frozen_skills)
            print(f"[SkillOnlyNoCode] Embedding retrieval indexed {len(frozen_skills)} skills")

    if not use_embedding:
        frozen_sm = SkillManager(
            library_path=skill_library_path or "",
            max_skills=skill_cfg.get("max_skills", 200),
            similarity_threshold=skill_cfg.get("similarity_threshold", 0.7),
        )
        frozen_sm.skills = list(frozen_skills)

    sem = asyncio.Semaphore(concurrency)
    results = []

    async def _one(item):
        question_text = benchmark.format_question(item)
        ground_truth = benchmark.get_ground_truth(item)

        tracker = UsageTracker()
        set_current_tracker(tracker)
        async with sem:
            # Retrieve skills
            retrieval_score_details = []
            if frozen_skills:
                if use_embedding:
                    retrieved_ws = await retriever.retrieve_with_scores(
                        question_text, top_k=skill_top_k,
                        threshold=skill_cfg.get("similarity_threshold", 0.3))
                    retrieval_score_details, _, _ = _extract_retrieval_scores(retrieved_ws)
                    retrieved = [s for s, _ in retrieved_ws]
                else:
                    retrieved = frozen_sm.retrieve(question_text, top_k=skill_top_k)
            else:
                retrieved = []

            # LLM relevance filter: use stronger model for better judgment
            skill_filter_reason = ""
            if retrieved:
                retrieved, skill_filter_reason = await _llm_filter_skills(
                    filter_llm, question_text, retrieved)

            # Create agent with retrieved skills, force_internal (no explore), no code
            temp_lib = os.path.join(tempfile.gettempdir(), f"eval_bm_sonc_{id(object())}.json")
            sm = SkillManager(
                library_path=temp_lib,
                max_skills=skill_cfg.get("max_skills", 200),
                similarity_threshold=skill_cfg.get("similarity_threshold", 0.7),
            )
            sm.skills = [normalize_skill(s) for s in retrieved]

            agent = ExplorationAgent(
                llm=student_llm,
                skill_manager=sm,
                web_search=_NoopWebSearch(),
                web_visit=_NoopWebVisit(),
                code_executor=None,
                max_rounds=max_rounds,
                explore_max_searches=0,
                force_internal=True,
                fetch_top_k=0,
                skip_skill_generation=True,
            )

            t0 = time.time()
            try:
                agent_result = await agent.solve(question_text, context=item.get("context", ""))
                answer = agent_result.get("answer", "")
                num_rounds = agent_result.get("num_rounds", 0)
                conversation = agent_result.get("conversation_history", [])
                called_skill_names = agent_result.get("called_skill_names", [])
                actions = agent_result.get("actions", [])
            except Exception as e:
                print(f"  [skill_only_no_code] [Error] {item['id']}: {e}")
                answer, num_rounds, conversation = "", 0, []
                called_skill_names = []
                actions = []
            elapsed = time.time() - t0
        usage = tracker.snapshot()

        # Check if any retrieved skill was actually called and count calls
        retrieved_skill_names_norm = {
            _normalize_skill_name(normalize_skill(s).get("skill_name", ""))
            for s in retrieved
        }
        retrieved_skill_used = bool(
            retrieved and any(n in retrieved_skill_names_norm for n in called_skill_names)
        )
        skill_call_count = sum(1 for n in called_skill_names if n in retrieved_skill_names_norm)
        called_retrieved_names = [n for n in called_skill_names if n in retrieved_skill_names_norm]

        judge_t0 = time.time()
        eval_result = await _async_evaluate(benchmark, answer, ground_truth,
                                             question=question_text)
        judge_elapsed = time.time() - judge_t0

        if os.path.exists(temp_lib):
            try:
                os.remove(temp_lib)
            except OSError:
                pass

        return {
            "id": item["id"],
            "domain": item.get("domain", ""),
            "difficulty": item.get("metadata", {}).get("difficulty", ""),
            "subfield": item.get("metadata", {}).get("subfield", ""),
            "question": item["question"][:200],
            "ground_truth": ground_truth,
            "prediction": answer[:500],
            "correct": eval_result["correct"],
            "score": eval_result["score"],
            "detail": eval_result["detail"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "num_llm_calls": usage["num_calls"],
            "web_search_count": 0,
            "num_rounds": num_rounds,
            "skills_retrieved": len(retrieved),
            "retrieved_skill_used": retrieved_skill_used,
            "skill_call_count": skill_call_count,
            "called_skill_names": called_retrieved_names,
            "retrieved_skill_scores": retrieval_score_details,
            "filter_kept_skill_names": [
                normalize_skill(s).get("skill_name", "") for s in retrieved
            ],
            "skill_filter_reason": skill_filter_reason,
            "conversation": conversation,
            "elapsed_sec": round(elapsed, 3),
        }

    tasks = [asyncio.create_task(_one(item)) for item in items]
    done_count = 0
    correct_count = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        if rt_writer:
            rt_writer.write_one(result)
        done_count += 1
        if result["correct"]:
            correct_count += 1
        acc = correct_count / done_count * 100
        sym = "+" if result["correct"] else "-"
        retr_tag = "R" if result["retrieved_skill_used"] else "."
        print(f"  [skill_only_no_code] [{done_count}/{len(items)}] {sym}{retr_tag}  "
              f"acc={acc:.1f}%  rounds={result['num_rounds']}  "
              f"retrieved={result['skills_retrieved']}  called={result['skill_call_count']}  "
              f"tokens={result['total_tokens']}")

    return results


# ---------------------------------------------------------------------------
# Mode: Free Tool -- direct tool access (query, page_visit, code) without EXPLORE gating
# ---------------------------------------------------------------------------

async def eval_free_tool(items, benchmark, config: dict, model_profile: str,
                         concurrency: int,
                         max_rounds: int,
                         rt_writer: Optional[RealtimeWriter] = None) -> List[Dict]:
    """Agent with direct tool access (query, page_visit, code) — no exploration trigger needed."""
    from core.free_tool_agent import FreeToolAgent
    from tools.web_search import WebSearchTool
    from tools.web_visit import WebVisitTool
    from tools.code_executor import CodeExecutor

    student_llm = LLMClient.from_config(config, profile_name=model_profile)
    attach_usage_tracking(student_llm)

    wv_cfg = config.get("web_visit", {})
    compress_profile = wv_cfg.get("compress_llm", model_profile)
    all_profiles = list(config.get("llm", {}).get("profiles", {}).keys())
    if compress_profile not in all_profiles:
        compress_profile = model_profile
    compress_llm = LLMClient.from_config(config, profile_name=compress_profile)
    attach_usage_tracking(compress_llm)

    web_search = WebSearchTool.from_config(config)
    web_visit = WebVisitTool.from_config(config, compress_llm=compress_llm)
    code_executor = CodeExecutor.from_config(config)

    agent_cfg = config["agent"]

    def _make_agent():
        return FreeToolAgent(
            llm=student_llm,
            web_search=web_search,
            web_visit=web_visit,
            code_executor=code_executor,
            max_rounds=max_rounds,
            fetch_top_k=agent_cfg.get("fetch_top_k", 3),
            explore_max_searches=agent_cfg.get("explore_max_searches", 5),
        )

    sem = asyncio.Semaphore(concurrency)
    results = []

    async def _one(item):
        question_text = benchmark.format_question(item)
        ground_truth = benchmark.get_ground_truth(item)

        tracker = UsageTracker()
        set_current_tracker(tracker)
        async with sem:
            agent = _make_agent()
            t0 = time.time()
            try:
                agent_result = await agent.solve(question_text, context=item.get("context", ""))
                answer = agent_result.get("answer", "")
                num_rounds = agent_result.get("num_rounds", 0)
                search_queries = agent_result.get("search_queries", [])
                conversation = agent_result.get("conversation_history", [])
                actions = agent_result.get("actions", [])
            except Exception as e:
                print(f"  [free_tool] [Error] {item['id']}: {e}")
                answer, num_rounds, search_queries, conversation = "", 0, [], []
                actions = []
            elapsed = time.time() - t0
        usage = tracker.snapshot()

        eval_result = await _async_evaluate(benchmark, answer, ground_truth,
                                             question=question_text)

        num_queries = sum(1 for a in actions if a == "query")
        num_page_visits = sum(1 for a in actions if a == "page_visit")

        return {
            "id": item["id"],
            "domain": item.get("domain", ""),
            "difficulty": item.get("metadata", {}).get("difficulty", ""),
            "subfield": item.get("metadata", {}).get("subfield", ""),
            "question": item["question"][:200],
            "ground_truth": ground_truth,
            "prediction": answer[:500],
            "correct": eval_result["correct"],
            "score": eval_result["score"],
            "detail": eval_result["detail"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "num_llm_calls": usage["num_calls"],
            "web_search_count": len(search_queries),
            "search_queries": search_queries,
            "num_rounds": num_rounds,
            "explore_triggered": num_queries > 0 or num_page_visits > 0,
            "num_queries": num_queries,
            "num_page_visits": num_page_visits,
            "conversation": conversation,
        }

    tasks = [asyncio.create_task(_one(item)) for item in items]
    done_count = 0
    correct_count = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        if rt_writer:
            rt_writer.write_one(result)
        done_count += 1
        if result["correct"]:
            correct_count += 1
        acc = correct_count / done_count * 100
        sym = "+" if result["correct"] else "-"
        print(f"  [free_tool] [{done_count}/{len(items)}] {sym}  "
              f"acc={acc:.1f}%  rounds={result['num_rounds']}  "
              f"queries={result['num_queries']}  tokens={result['total_tokens']}")

    # Aggregate stats
    total_queries = sum(r.get("num_queries", 0) for r in results)
    total_pv = sum(r.get("num_page_visits", 0) for r in results)
    avg_q = total_queries / len(results) if results else 0.0
    avg_pv = total_pv / len(results) if results else 0.0
    print(f"[FreeTool] avg_queries={avg_q:.2f}, avg_page_visits={avg_pv:.2f}")

    return results


# ---------------------------------------------------------------------------
# Mode 4: Search + Skill -- agent + web + code + skill gen + retrieval
# ---------------------------------------------------------------------------

def _normalize_skill_name(value: str) -> str:
    """Normalize skill name for comparison (mirrors ExplorationAgent._normalize_skill_name)."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _extract_retrieval_scores(retrieved_with_scores):
    """Build score fields from retrieve_with_scores() output (List[(skill, score)])."""
    if not retrieved_with_scores:
        return [], None, None
    details = [{"name": s.get("skill_name", ""), "score": sc}
               for s, sc in retrieved_with_scores]
    scores = [sc for _, sc in retrieved_with_scores]
    return details, max(scores), min(scores)


async def eval_search_skill(items, benchmark, config: dict, model_profile: str,
                            concurrency: int, max_rounds: int,
                            skill_library_path: Optional[str],
                            skill_top_k: int,
                            retrieval: str = "embedding",
                            batch_size: int = 10,
                            rt_writer: Optional[RealtimeWriter] = None,
                            resume: bool = False,
                            disable_retrieval: bool = False,
                            force_first_explore: bool = False) -> List[Dict]:
    """Agent + web search + code + persistent skills + retrieval.

    Args:
        retrieval: "fuzzy" for SkillManager.retrieve() (rapidfuzz),
                   "embedding" for EmbeddingRetriever (Qwen3-Embedding).
        batch_size: Number of questions per wave. Questions within a wave run
                    concurrently sharing the same skill library snapshot; skills
                    accumulate across waves. Set 1 for fully sequential.
    """
    from core.agent import ExplorationAgent
    from core.skill_manager import SkillManager
    from core.skill_schema import normalize_skill
    from tools.web_search import WebSearchTool
    from tools.web_visit import WebVisitTool
    from tools.code_executor import CodeExecutor

    student_llm = LLMClient.from_config(config, profile_name=model_profile)
    attach_usage_tracking(student_llm)

    wv_cfg = config.get("web_visit", {})
    compress_profile = wv_cfg.get("compress_llm", model_profile)
    all_profiles = list(config.get("llm", {}).get("profiles", {}).keys())
    if compress_profile not in all_profiles:
        compress_profile = model_profile
    compress_llm = LLMClient.from_config(config, profile_name=compress_profile)
    attach_usage_tracking(compress_llm)

    web_search = WebSearchTool.from_config(config)
    web_visit = WebVisitTool.from_config(config, compress_llm=compress_llm)
    code_executor = CodeExecutor.from_config(config)

    skill_cfg = config.get("skill", {})
    agent_cfg = config["agent"]
    agent_cfg["max_rounds"] = max_rounds

    # Persistent skill library — clear on re-run to start fresh (unless resuming)
    if skill_library_path is None:
        skill_library_path = os.path.join(
            tempfile.gettempdir(),
            f"eval_bm_skill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    if not resume and os.path.exists(skill_library_path):
        os.remove(skill_library_path)
    persistent_sm = SkillManager(
        library_path=skill_library_path,
        max_skills=skill_cfg.get("max_skills", 200),
        similarity_threshold=skill_cfg.get("similarity_threshold", 0.7),
    )

    # All-skill library (correct + incorrect, for logging only)
    all_skill_library_path = skill_library_path.replace(".json", "_all.json")
    if not resume and os.path.exists(all_skill_library_path):
        os.remove(all_skill_library_path)
    all_skill_sm = SkillManager(
        library_path=all_skill_library_path,
        max_skills=skill_cfg.get("max_skills", 200),
        similarity_threshold=skill_cfg.get("similarity_threshold", 0.7),
    )

    # Embedding retriever (only for retrieval="embedding")
    use_embedding = False
    retriever = None
    if retrieval == "embedding":
        from evaluation.embedding_retriever import EmbeddingRetriever
        retriever = EmbeddingRetriever.from_config(config)
        if retriever:
            use_embedding = True
            if persistent_sm.skills:
                await retriever.index_skills(persistent_sm.skills)
            print(f"[SearchSkill] Embedding retrieval (model: {retriever.model}), "
                  f"top_k={skill_top_k}, library={skill_library_path}")
        else:
            print("[SearchSkill] No embedding config found, falling back to fuzzy retrieval")
            retrieval = "fuzzy"

    if not use_embedding:
        print(f"[SearchSkill] Fuzzy retrieval, top_k={skill_top_k}, "
              f"library={skill_library_path}")

    filter_llm = _get_filter_llm(config, model_profile)

    sem = asyncio.Semaphore(concurrency)
    results = []
    done_count = 0
    correct_count = 0

    async def _solve_one(item, retrieved_skills, retrieval_score_details=None):
        """Solve a single question with pre-retrieved skills."""
        if retrieval_score_details is None:
            retrieval_score_details = []
        question_text = benchmark.format_question(item)
        ground_truth = benchmark.get_ground_truth(item)

        temp_lib = os.path.join(tempfile.gettempdir(), f"eval_bm_ss_q_{id(object())}.json")
        sm = SkillManager(
            library_path=temp_lib,
            max_skills=skill_cfg.get("max_skills", 200),
            similarity_threshold=skill_cfg.get("similarity_threshold", 0.7),
        )
        sm.skills = [normalize_skill(s) for s in retrieved_skills]

        agent = ExplorationAgent(
            llm=student_llm,
            skill_manager=sm,
            web_search=web_search,
            web_visit=web_visit,
            code_executor=code_executor,
            max_rounds=max_rounds,
            explore_max_searches=agent_cfg.get("explore_max_searches", 5),
            force_internal=False,
            fetch_top_k=agent_cfg.get("fetch_top_k", 3),
            force_first_explore=force_first_explore,
        )

        tracker = UsageTracker()
        set_current_tracker(tracker)
        async with sem:
            t0 = time.time()
            try:
                agent_result = await agent.solve(question_text, context=item.get("context", ""))
                answer = agent_result.get("answer", "")
                num_rounds = agent_result.get("num_rounds", 0)
                search_queries = agent_result.get("search_queries", [])
                skills_generated = agent_result.get("skills_generated_all", [])
                conversation = agent_result.get("conversation_history", [])
                called_skill_names = agent_result.get("called_skill_names", [])
                actions = agent_result.get("actions", [])
                profile_stats = agent_result.get("profile_stats", {})
            except Exception as e:
                print(f"  [search_skill] [Error] {item['id']}: {e}")
                answer, num_rounds, search_queries, skills_generated, conversation = "", 0, [], [], []
                called_skill_names = []
                actions = []
                profile_stats = {}
            elapsed = time.time() - t0
        usage = tracker.snapshot()

        # Check if any retrieved skill was actually called
        retrieved_skill_names_norm = {
            _normalize_skill_name(normalize_skill(s).get("skill_name", ""))
            for s in retrieved_skills
        }
        retrieved_skill_used = bool(
            retrieved_skills
            and any(n in retrieved_skill_names_norm for n in called_skill_names)
        )

        judge_t0 = time.time()
        eval_result = await _async_evaluate(benchmark, answer, ground_truth,
                                             question=question_text)
        judge_elapsed = time.time() - judge_t0

        # Cleanup temp library
        if os.path.exists(temp_lib):
            try:
                os.remove(temp_lib)
            except OSError:
                pass

        return {
            "id": item["id"],
            "domain": item.get("domain", ""),
            "difficulty": item.get("metadata", {}).get("difficulty", ""),
            "subfield": item.get("metadata", {}).get("subfield", ""),
            "question": item["question"][:200],
            "ground_truth": ground_truth,
            "prediction": answer[:500],
            "correct": eval_result["correct"],
            "score": eval_result["score"],
            "detail": eval_result["detail"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "num_llm_calls": usage["num_calls"],
            "web_search_count": len(search_queries),
            "search_queries": search_queries,
            "num_rounds": num_rounds,
            "skills_generated": len(skills_generated),
            "skills_in_library": 0,  # updated after wave completes
            "skills_retrieved": len(retrieved_skills),
            "retrieved_skill_used": retrieved_skill_used,
            "retrieved_skill_scores": retrieval_score_details,
            "explore_triggered": any(a == "explore" for a in actions),
            "num_queries": sum(1 for a in actions if a == "query"),
            "num_page_visits": sum(1 for a in actions if a == "page_visit"),
            "conversation": conversation,
            "elapsed_sec": round(elapsed, 3),
            "llm_time_sec": round(profile_stats.get("llm_time_sec", 0.0), 3),
            "search_time_sec": round(profile_stats.get("search_time_sec", 0.0), 3),
            "page_visit_time_sec": round(profile_stats.get("page_visit_time_sec", 0.0), 3),
            "code_time_sec": round(profile_stats.get("code_time_sec", 0.0), 3),
            "teacher_rewrite_time_sec": round(profile_stats.get("teacher_rewrite_time_sec", 0.0), 3),
            "ensure_answer_time_sec": round(profile_stats.get("ensure_answer_time_sec", 0.0), 3),
            "judge_time_sec": round(judge_elapsed, 3),
            "_skills_generated_raw": skills_generated,  # temp, for persisting after wave
        }

    # ---- Streaming execution: maintain `concurrency` in-flight tasks ----
    # As each task completes, persist skills immediately and launch next question.
    # New questions retrieve from the up-to-date skill library.

    item_queue = list(range(len(items)))  # indices of items to process
    next_idx = 0  # next item to launch
    in_flight: dict[asyncio.Task, int] = {}  # task -> item_index

    async def _launch_one(idx: int) -> asyncio.Task:
        """Retrieve skills and launch a solve task for items[idx]."""
        item = items[idx]
        retrieval_score_details = []
        if not disable_retrieval and persistent_sm.skills:
            question_text = benchmark.format_question(item)
            if use_embedding:
                retrieved_ws = await retriever.retrieve_with_scores(
                    question_text, top_k=skill_top_k,
                    threshold=skill_cfg.get("similarity_threshold", 0.3))
                retrieval_score_details, _, _ = _extract_retrieval_scores(retrieved_ws)
                retrieved = [s for s, _ in retrieved_ws]
            else:
                retrieved = persistent_sm.retrieve(question_text, top_k=skill_top_k)
            if retrieved and filter_llm:
                retrieved, _ = await _llm_filter_skills(filter_llm, question_text, retrieved)
        else:
            retrieved = []
        task = asyncio.create_task(_solve_one(item, retrieved, retrieval_score_details))
        return task

    # Launch initial batch
    while next_idx < len(items) and len(in_flight) < concurrency:
        task = await _launch_one(next_idx)
        in_flight[task] = next_idx
        next_idx += 1

    print(f"[SearchSkill] Streaming mode: concurrency={concurrency}, "
          f"{len(items)} questions, library starts with {len(persistent_sm.skills)} skills")

    while in_flight:
        # Wait for any one task to complete
        done_set, _ = await asyncio.wait(
            in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
        )

        for task in done_set:
            item_idx = in_flight.pop(task)
            result = task.result()

            # Persist generated skills: all to both libraries
            generated_skills = result.pop("_skills_generated_raw", [])
            for new_skill in generated_skills:
                ns = normalize_skill(new_skill)
                if ns.get("skill_id") and not ns.get("builtin"):
                    ns_log = dict(ns)
                    ns_log["from_correct"] = result["correct"]
                    existing_all = all_skill_sm.get_skill_by_name(ns.get("skill_name", ""))
                    if not existing_all:
                        try:
                            assigned_id = all_skill_sm.add_skill(ns_log)
                        except Exception:
                            assigned_id = None
                    else:
                        assigned_id = existing_all.get("skill_id")

                    existing = persistent_sm.get_skill_by_name(ns.get("skill_name", ""))
                    if not existing:
                        try:
                            if assigned_id:
                                ns["skill_id"] = assigned_id
                            persistent_sm.add_skill(ns)
                            if use_embedding:
                                await retriever.add_skill_auto(ns)
                        except Exception as e:
                            print(f"  [search_skill] [Warn] Failed to persist skill: {e}")

            result["skills_in_library"] = len(persistent_sm.skills)
            results.append(result)
            if rt_writer:
                rt_writer.write_one(result)
            done_count += 1
            if result["correct"]:
                correct_count += 1
            acc = correct_count / done_count * 100
            sym = "+" if result["correct"] else "-"
            retr_tag = "R" if result["retrieved_skill_used"] else "."
            exp_tag = "E" if result["explore_triggered"] else "."
            print(f"  [search_skill] [{done_count}/{len(items)}] {sym}{retr_tag}{exp_tag}  "
                  f"acc={acc:.1f}%  rounds={result['num_rounds']}  "
                  f"queries={result['num_queries']}  gen={result['skills_generated']}  "
                  f"skills={len(persistent_sm.skills)}({len(all_skill_sm.skills)})  "
                  f"tokens={result['total_tokens']}  "
                  f"t={result.get('elapsed_sec', 0):.1f}s llm={result.get('llm_time_sec', 0):.1f}s "
                  f"search={result.get('search_time_sec', 0):.1f}s visit={result.get('page_visit_time_sec', 0):.1f}s "
                  f"code={result.get('code_time_sec', 0):.1f}s judge={result.get('judge_time_sec', 0):.1f}s")

            # Launch next question if available
            if next_idx < len(items):
                new_task = await _launch_one(next_idx)
                in_flight[new_task] = next_idx
                next_idx += 1

    # Aggregate retrieved-skill trigger rate
    has_retrieved = [r for r in results if r["skills_retrieved"] > 0]
    triggered = [r for r in has_retrieved if r["retrieved_skill_used"]]
    trigger_rate = len(triggered) / len(has_retrieved) * 100 if has_retrieved else 0.0
    print(f"[SearchSkill] Retrieved skill trigger rate: "
          f"{len(triggered)}/{len(has_retrieved)} = {trigger_rate:.1f}% "
          f"(out of questions with retrieved skills)")
    # Aggregate exploration stats
    explored = [r for r in results if r.get("explore_triggered")]
    explore_rate = len(explored) / len(results) * 100 if results else 0.0
    total_queries = sum(r.get("num_queries", 0) for r in explored)
    total_pv = sum(r.get("num_page_visits", 0) for r in explored)
    avg_q = total_queries / len(explored) if explored else 0.0
    avg_pv = total_pv / len(explored) if explored else 0.0
    total_gen = sum(r.get("skills_generated", 0) for r in results)
    print(f"[SearchSkill] Explore rate: {len(explored)}/{len(results)} = {explore_rate:.1f}%")
    print(f"[SearchSkill] Among explored: avg_queries={avg_q:.2f}, avg_page_visits={avg_pv:.2f}")
    print(f"[SearchSkill] Total skills generated: {total_gen}")
    print(f"[SearchSkill] Final skill library: {len(persistent_sm.skills)} correct-only skills "
          f"at {skill_library_path}")
    print(f"[SearchSkill] All-skill library: {len(all_skill_sm.skills)} total skills "
          f"at {all_skill_library_path}")
    all_skill_sm.save_library()
    return results


# ---------------------------------------------------------------------------
# Mode 4b: Search + Skill (No Code) -- identical to search_skill but
#           without code execution tool
# ---------------------------------------------------------------------------

async def eval_search_skill_no_code(items, benchmark, config: dict, model_profile: str,
                            concurrency: int, max_rounds: int,
                            skill_library_path: Optional[str],
                            skill_top_k: int,
                            retrieval: str = "embedding",
                            batch_size: int = 10,
                            rt_writer: Optional[RealtimeWriter] = None,
                            resume: bool = False,
                            disable_retrieval: bool = False) -> List[Dict]:
    """Agent + web search + persistent skills + retrieval, WITHOUT code execution."""
    from core.agent import ExplorationAgent
    from core.skill_manager import SkillManager
    from core.skill_schema import normalize_skill
    from tools.web_search import WebSearchTool
    from tools.web_visit import WebVisitTool

    student_llm = LLMClient.from_config(config, profile_name=model_profile)
    attach_usage_tracking(student_llm)

    wv_cfg = config.get("web_visit", {})
    compress_profile = wv_cfg.get("compress_llm", model_profile)
    all_profiles = list(config.get("llm", {}).get("profiles", {}).keys())
    if compress_profile not in all_profiles:
        compress_profile = model_profile
    compress_llm = LLMClient.from_config(config, profile_name=compress_profile)
    attach_usage_tracking(compress_llm)

    web_search = WebSearchTool.from_config(config)
    web_visit = WebVisitTool.from_config(config, compress_llm=compress_llm)

    skill_cfg = config.get("skill", {})
    agent_cfg = config["agent"]
    agent_cfg["max_rounds"] = max_rounds

    if skill_library_path is None:
        skill_library_path = os.path.join(
            tempfile.gettempdir(),
            f"eval_bm_skill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    if not resume and os.path.exists(skill_library_path):
        os.remove(skill_library_path)
    persistent_sm = SkillManager(
        library_path=skill_library_path,
        max_skills=skill_cfg.get("max_skills", 200),
        similarity_threshold=skill_cfg.get("similarity_threshold", 0.7),
    )

    # All-skill library (correct + incorrect, for logging only)
    all_skill_library_path = skill_library_path.replace(".json", "_all.json")
    if not resume and os.path.exists(all_skill_library_path):
        os.remove(all_skill_library_path)
    all_skill_sm = SkillManager(
        library_path=all_skill_library_path,
        max_skills=skill_cfg.get("max_skills", 200),
        similarity_threshold=skill_cfg.get("similarity_threshold", 0.7),
    )

    use_embedding = False
    retriever = None
    if retrieval == "embedding":
        from evaluation.embedding_retriever import EmbeddingRetriever
        retriever = EmbeddingRetriever.from_config(config)
        if retriever:
            use_embedding = True
            if persistent_sm.skills:
                await retriever.index_skills(persistent_sm.skills)
            print(f"[SearchSkillNoCode] Embedding retrieval (model: {retriever.model}), "
                  f"top_k={skill_top_k}, library={skill_library_path}")
        else:
            print("[SearchSkillNoCode] No embedding config found, falling back to fuzzy retrieval")
            retrieval = "fuzzy"

    if not use_embedding:
        print(f"[SearchSkillNoCode] Fuzzy retrieval, top_k={skill_top_k}, "
              f"library={skill_library_path}")

    filter_llm = _get_filter_llm(config, model_profile)

    sem = asyncio.Semaphore(concurrency)
    results = []
    done_count = 0
    correct_count = 0

    async def _solve_one(item, retrieved_skills, retrieval_score_details=None):
        if retrieval_score_details is None:
            retrieval_score_details = []
        question_text = benchmark.format_question(item)
        ground_truth = benchmark.get_ground_truth(item)

        temp_lib = os.path.join(tempfile.gettempdir(), f"eval_bm_ssnc_q_{id(object())}.json")
        sm = SkillManager(
            library_path=temp_lib,
            max_skills=skill_cfg.get("max_skills", 200),
            similarity_threshold=skill_cfg.get("similarity_threshold", 0.7),
        )
        sm.skills = [normalize_skill(s) for s in retrieved_skills]

        agent = ExplorationAgent(
            llm=student_llm,
            skill_manager=sm,
            web_search=web_search,
            web_visit=web_visit,
            code_executor=None,
            max_rounds=max_rounds,
            explore_max_searches=agent_cfg.get("explore_max_searches", 5),
            force_internal=False,
            fetch_top_k=agent_cfg.get("fetch_top_k", 3),
        )

        tracker = UsageTracker()
        set_current_tracker(tracker)
        async with sem:
            t0 = time.time()
            try:
                agent_result = await agent.solve(question_text, context=item.get("context", ""))
                answer = agent_result.get("answer", "")
                num_rounds = agent_result.get("num_rounds", 0)
                search_queries = agent_result.get("search_queries", [])
                skills_generated = agent_result.get("skills_generated_all", [])
                conversation = agent_result.get("conversation_history", [])
                called_skill_names = agent_result.get("called_skill_names", [])
                actions = agent_result.get("actions", [])
                profile_stats = agent_result.get("profile_stats", {})
            except Exception as e:
                print(f"  [search_skill_no_code] [Error] {item['id']}: {e}")
                answer, num_rounds, search_queries, skills_generated, conversation = "", 0, [], [], []
                called_skill_names = []
                actions = []
                profile_stats = {}
            elapsed = time.time() - t0
        usage = tracker.snapshot()

        retrieved_skill_names_norm = {
            _normalize_skill_name(normalize_skill(s).get("skill_name", ""))
            for s in retrieved_skills
        }
        retrieved_skill_used = bool(
            retrieved_skills
            and any(n in retrieved_skill_names_norm for n in called_skill_names)
        )

        judge_t0 = time.time()
        eval_result = await _async_evaluate(benchmark, answer, ground_truth,
                                             question=question_text)
        judge_elapsed = time.time() - judge_t0

        if os.path.exists(temp_lib):
            try:
                os.remove(temp_lib)
            except OSError:
                pass

        return {
            "id": item["id"],
            "domain": item.get("domain", ""),
            "difficulty": item.get("metadata", {}).get("difficulty", ""),
            "subfield": item.get("metadata", {}).get("subfield", ""),
            "question": item["question"][:200],
            "ground_truth": ground_truth,
            "prediction": answer[:500],
            "correct": eval_result["correct"],
            "score": eval_result["score"],
            "detail": eval_result["detail"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "num_llm_calls": usage["num_calls"],
            "web_search_count": len(search_queries),
            "search_queries": search_queries,
            "num_rounds": num_rounds,
            "skills_generated": len(skills_generated),
            "skills_in_library": 0,
            "skills_retrieved": len(retrieved_skills),
            "retrieved_skill_used": retrieved_skill_used,
            "retrieved_skill_scores": retrieval_score_details,
            "explore_triggered": any(a == "explore" for a in actions),
            "num_queries": sum(1 for a in actions if a == "query"),
            "num_page_visits": sum(1 for a in actions if a == "page_visit"),
            "conversation": conversation,
            "elapsed_sec": round(elapsed, 3),
            "llm_time_sec": round(profile_stats.get("llm_time_sec", 0.0), 3),
            "search_time_sec": round(profile_stats.get("search_time_sec", 0.0), 3),
            "page_visit_time_sec": round(profile_stats.get("page_visit_time_sec", 0.0), 3),
            "code_time_sec": 0.0,
            "teacher_rewrite_time_sec": round(profile_stats.get("teacher_rewrite_time_sec", 0.0), 3),
            "ensure_answer_time_sec": round(profile_stats.get("ensure_answer_time_sec", 0.0), 3),
            "judge_time_sec": round(judge_elapsed, 3),
            "_skills_generated_raw": skills_generated,
        }

    item_queue = list(range(len(items)))
    next_idx = 0
    in_flight: dict[asyncio.Task, int] = {}

    async def _launch_one(idx: int) -> asyncio.Task:
        """Retrieve skills and launch a solve task for items[idx]."""
        item = items[idx]
        retrieval_score_details = []
        if not disable_retrieval and persistent_sm.skills:
            question_text = benchmark.format_question(item)
            if use_embedding:
                retrieved_ws = await retriever.retrieve_with_scores(
                    question_text, top_k=skill_top_k,
                    threshold=skill_cfg.get("similarity_threshold", 0.3))
                retrieval_score_details, _, _ = _extract_retrieval_scores(retrieved_ws)
                retrieved = [s for s, _ in retrieved_ws]
            else:
                retrieved = persistent_sm.retrieve(question_text, top_k=skill_top_k)
            if retrieved and filter_llm:
                retrieved, _ = await _llm_filter_skills(filter_llm, question_text, retrieved)
        else:
            retrieved = []
        task = asyncio.create_task(_solve_one(item, retrieved, retrieval_score_details))
        return task

    while next_idx < len(items) and len(in_flight) < concurrency:
        task = await _launch_one(next_idx)
        in_flight[task] = next_idx
        next_idx += 1

    print(f"[SearchSkillNoCode] Streaming mode: concurrency={concurrency}, "
          f"{len(items)} questions, library starts with {len(persistent_sm.skills)} skills")

    while in_flight:
        done_set, _ = await asyncio.wait(
            in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
        )

        for task in done_set:
            item_idx = in_flight.pop(task)
            result = task.result()

            generated_skills = result.pop("_skills_generated_raw", [])
            for new_skill in generated_skills:
                ns = normalize_skill(new_skill)
                if ns.get("skill_id") and not ns.get("builtin"):
                    ns_log = dict(ns)
                    ns_log["from_correct"] = result["correct"]
                    existing_all = all_skill_sm.get_skill_by_name(ns.get("skill_name", ""))
                    if not existing_all:
                        try:
                            assigned_id = all_skill_sm.add_skill(ns_log)
                        except Exception:
                            assigned_id = None
                    else:
                        assigned_id = existing_all.get("skill_id")

                    existing = persistent_sm.get_skill_by_name(ns.get("skill_name", ""))
                    if not existing:
                        try:
                            if assigned_id:
                                ns["skill_id"] = assigned_id
                            persistent_sm.add_skill(ns)
                            if use_embedding:
                                await retriever.add_skill_auto(ns)
                        except Exception as e:
                            print(f"  [search_skill_no_code] [Warn] Failed to persist skill: {e}")

            result["skills_in_library"] = len(persistent_sm.skills)
            results.append(result)
            if rt_writer:
                rt_writer.write_one(result)
            done_count += 1
            if result["correct"]:
                correct_count += 1
            acc = correct_count / done_count * 100
            sym = "+" if result["correct"] else "-"
            retr_tag = "R" if result["retrieved_skill_used"] else "."
            exp_tag = "E" if result["explore_triggered"] else "."
            print(f"  [search_skill_no_code] [{done_count}/{len(items)}] {sym}{retr_tag}{exp_tag}  "
                  f"acc={acc:.1f}%  rounds={result['num_rounds']}  "
                  f"queries={result['num_queries']}  gen={result['skills_generated']}  "
                  f"skills={len(persistent_sm.skills)}({len(all_skill_sm.skills)})  "
                  f"tokens={result['total_tokens']}  "
                  f"t={result.get('elapsed_sec', 0):.1f}s llm={result.get('llm_time_sec', 0):.1f}s "
                  f"search={result.get('search_time_sec', 0):.1f}s visit={result.get('page_visit_time_sec', 0):.1f}s "
                  f"judge={result.get('judge_time_sec', 0):.1f}s")

            if next_idx < len(items):
                new_task = await _launch_one(next_idx)
                in_flight[new_task] = next_idx
                next_idx += 1

    has_retrieved = [r for r in results if r["skills_retrieved"] > 0]
    triggered = [r for r in has_retrieved if r["retrieved_skill_used"]]
    trigger_rate = len(triggered) / len(has_retrieved) * 100 if has_retrieved else 0.0
    print(f"[SearchSkillNoCode] Retrieved skill trigger rate: "
          f"{len(triggered)}/{len(has_retrieved)} = {trigger_rate:.1f}% "
          f"(out of questions with retrieved skills)")
    explored = [r for r in results if r.get("explore_triggered")]
    explore_rate = len(explored) / len(results) * 100 if results else 0.0
    total_queries = sum(r.get("num_queries", 0) for r in explored)
    total_pv = sum(r.get("num_page_visits", 0) for r in explored)
    avg_q = total_queries / len(explored) if explored else 0.0
    avg_pv = total_pv / len(explored) if explored else 0.0
    total_gen = sum(r.get("skills_generated", 0) for r in results)
    print(f"[SearchSkillNoCode] Explore rate: {len(explored)}/{len(results)} = {explore_rate:.1f}%")
    print(f"[SearchSkillNoCode] Among explored: avg_queries={avg_q:.2f}, avg_page_visits={avg_pv:.2f}")
    print(f"[SearchSkillNoCode] Total skills generated: {total_gen}")
    print(f"[SearchSkillNoCode] Final skill library: {len(persistent_sm.skills)} correct-only skills "
          f"at {skill_library_path}")
    print(f"[SearchSkillNoCode] All-skill library: {len(all_skill_sm.skills)} total skills "
          f"at {all_skill_library_path}")
    all_skill_sm.save_library()
    return results


# ---------------------------------------------------------------------------
# Summary & output (research-grade multi-dimensional breakdown)
# ---------------------------------------------------------------------------

def _agg(records: List[Dict]) -> Dict:
    """Aggregate a list of per-question result dicts into summary metrics."""
    n = len(records)
    if n == 0:
        return {"n": 0, "acc": 0.0, "avg_tokens": 0.0, "avg_prompt": 0.0,
                "avg_completion": 0.0, "avg_searches": 0.0, "avg_rounds": 0.0,
                "retrieved_skill_trigger": "N/A"}
    correct = sum(1 for r in records if r["correct"])
    agg = {
        "n": n,
        "correct": correct,
        "acc": round(correct / n, 4),
        "avg_tokens": round(sum(r.get("total_tokens", 0) for r in records) / n, 1),
        "avg_prompt": round(sum(r.get("prompt_tokens", 0) for r in records) / n, 1),
        "avg_completion": round(sum(r.get("completion_tokens", 0) for r in records) / n, 1),
        "avg_searches": round(sum(r.get("web_search_count", 0) for r in records) / n, 2),
        "avg_rounds": round(sum(r.get("num_rounds", 0) for r in records) / n, 2),
    }
    # Retrieved skill trigger rate (only meaningful for search_skill mode)
    has_retrieved = [r for r in records if r.get("skills_retrieved", 0) > 0]
    if has_retrieved:
        triggered = sum(1 for r in has_retrieved if r.get("retrieved_skill_used"))
        agg["retrieved_skill_trigger"] = f"{triggered}/{len(has_retrieved)}"
        agg["retrieved_skill_trigger_rate"] = round(triggered / len(has_retrieved), 4)
    else:
        agg["retrieved_skill_trigger"] = "N/A"
        agg["retrieved_skill_trigger_rate"] = None

    # Exploration stats (search_skill)
    explored = [r for r in records if r.get("explore_triggered")]
    if any("explore_triggered" in r for r in records):
        agg["explore_count"] = len(explored)
        agg["explore_rate"] = round(len(explored) / n, 4)
        total_queries = sum(r.get("num_queries", 0) for r in explored)
        total_pv = sum(r.get("num_page_visits", 0) for r in explored)
        agg["avg_queries_per_explore"] = round(total_queries / len(explored), 2) if explored else 0.0
        agg["avg_page_visits_per_explore"] = round(total_pv / len(explored), 2) if explored else 0.0
    else:
        agg["explore_count"] = None
        agg["explore_rate"] = None
        agg["avg_queries_per_explore"] = None
        agg["avg_page_visits_per_explore"] = None

    # Skill generation stats (search_skill mode)
    if any("skills_generated" in r for r in records):
        total_gen = sum(r.get("skills_generated", 0) for r in records)
        agg["total_skills_generated"] = total_gen
        agg["avg_skills_generated"] = round(total_gen / n, 2)
    else:
        agg["total_skills_generated"] = None
        agg["avg_skills_generated"] = None

    return agg


def _group_by(results: List[Dict], key: str) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for r in results:
        k = str(r.get(key, "") or "unknown")
        groups.setdefault(k, []).append(r)
    return groups


def compute_summary(results: List[Dict], args) -> Dict:
    """Compute multi-dimensional aggregated summary."""
    overall = _agg(results)

    # By discipline (domain)
    by_domain = {k: _agg(v) for k, v in _group_by(results, "domain").items()}

    # By difficulty
    by_difficulty = {k: _agg(v) for k, v in _group_by(results, "difficulty").items()}

    # By domain x difficulty
    by_domain_difficulty = {}
    for domain, domain_records in _group_by(results, "domain").items():
        by_domain_difficulty[domain] = {
            k: _agg(v) for k, v in _group_by(domain_records, "difficulty").items()
        }

    return {
        "_summary": True,
        "benchmark": args.benchmark,
        "mode": args.mode,
        "model": args.model,
        "subset": args.subset,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "max_rounds": args.max_rounds,
        "overall": overall,
        "by_discipline": by_domain,
        "by_difficulty": by_difficulty,
        "by_discipline_x_difficulty": by_domain_difficulty,
    }


# ---- ASCII table helpers ----

def _fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}"


def _ascii_table(headers: List[str], rows: List[List[str]], min_col: int = 10) -> str:
    """Render a simple ASCII table."""
    widths = [max(min_col, len(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def _row_str(cells):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [_row_str(headers), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    for row in rows:
        lines.append(_row_str(row))
    return "\n".join(lines)


def print_summary(summary: Dict):
    """Print research-grade summary tables to stdout."""
    o = summary["overall"]
    w = 72
    print(f"\n{'=' * w}")
    print(f"  {summary.get('benchmark', '?')}  |  mode={summary['mode']}  |  "
          f"model={summary.get('model', '?')}")
    print(f"{'=' * w}")
    print(f"  Overall: {o['correct']}/{o['n']} = {_fmt_pct(o['acc'])}%")
    print(f"  Avg tokens/q:     {o['avg_tokens']:.1f}  "
          f"(prompt={o['avg_prompt']:.1f}, completion={o['avg_completion']:.1f})")
    print(f"  Avg web search/q: {o['avg_searches']:.2f}")
    print(f"  Avg rounds/q:     {o['avg_rounds']:.2f}")
    if o.get("retrieved_skill_trigger") and o["retrieved_skill_trigger"] != "N/A":
        rate = o.get("retrieved_skill_trigger_rate", 0) or 0
        print(f"  Retrieved skill trigger: {o['retrieved_skill_trigger']} = {_fmt_pct(rate)}%")
    if o.get("explore_rate") is not None:
        print(f"  Explore rate:     {o['explore_count']}/{o['n']} = {_fmt_pct(o['explore_rate'])}%")
        print(f"  Per explored q:   avg_queries={o['avg_queries_per_explore']:.2f}, "
              f"avg_page_visits={o['avg_page_visits_per_explore']:.2f}")
    if o.get("total_skills_generated") is not None:
        print(f"  Skills generated: {o['total_skills_generated']} total, "
              f"{o['avg_skills_generated']:.2f}/q avg")
    if o.get("total_teacher_rewrites") is not None:
        print(f"  Teacher rewrites: {o['total_teacher_rewrites']} total "
              f"(on {o['teacher_rewrite_questions']}/{o['n']} questions), "
              f"{o['avg_teacher_rewrites']:.2f}/q avg")
    print(f"{'=' * w}")

    # -- Table 1: By Discipline --
    by_d = summary.get("by_discipline", {})
    has_trigger = any(
        s.get("retrieved_skill_trigger", "N/A") != "N/A"
        for s in by_d.values()
    )
    if by_d:
        print(f"\n  [By Discipline]")
        headers = ["Discipline", "N", "Correct", "Acc%", "AvgTokens", "AvgSearch", "AvgRounds"]
        if has_trigger:
            headers.append("SkillTrigger")
        rows = []
        for d in sorted(by_d.keys()):
            s = by_d[d]
            row = [d, str(s["n"]), str(s["correct"]), _fmt_pct(s["acc"]),
                   f"{s['avg_tokens']:.0f}", f"{s['avg_searches']:.1f}",
                   f"{s['avg_rounds']:.1f}"]
            if has_trigger:
                tr = s.get("retrieved_skill_trigger", "N/A")
                rate = s.get("retrieved_skill_trigger_rate")
                row.append(f"{tr} ({_fmt_pct(rate)}%)" if rate is not None else tr)
            rows.append(row)
        print(_ascii_table(headers, rows))

    # -- Table 2: By Difficulty --
    by_diff = summary.get("by_difficulty", {})
    if by_diff and len(by_diff) > 1:
        print(f"\n  [By Difficulty]")
        headers = ["Difficulty", "N", "Correct", "Acc%", "AvgTokens", "AvgSearch", "AvgRounds"]
        rows = []
        for d in sorted(by_diff.keys()):
            s = by_diff[d]
            rows.append([d, str(s["n"]), str(s["correct"]), _fmt_pct(s["acc"]),
                         f"{s['avg_tokens']:.0f}", f"{s['avg_searches']:.1f}",
                         f"{s['avg_rounds']:.1f}"])
        print(_ascii_table(headers, rows))

    # -- Table 3: Discipline x Difficulty (compact pivot) --
    by_dd = summary.get("by_discipline_x_difficulty", {})
    if by_dd and by_diff and len(by_diff) > 1:
        all_diffs = sorted({d for dd in by_dd.values() for d in dd.keys()})
        if len(all_diffs) <= 10:
            print(f"\n  [Discipline x Difficulty — Accuracy%]")
            headers = ["Discipline"] + all_diffs + ["Overall"]
            rows = []
            for domain in sorted(by_dd.keys()):
                row = [domain]
                for diff in all_diffs:
                    s = by_dd[domain].get(diff)
                    if s:
                        row.append(f"{_fmt_pct(s['acc'])} (N={s['n']})")
                    else:
                        row.append("-")
                ds = by_d.get(domain, {})
                row.append(f"{_fmt_pct(ds.get('acc', 0))} (N={ds.get('n', 0)})")
                rows.append(row)
            print(_ascii_table(headers, rows, min_col=8))

    print(f"{'=' * w}")


def save_results(results: List[Dict], summary: Dict, output_dir: str, mode: str, args,
                 rt_writer: Optional[RealtimeWriter] = None):
    """Save results into {output_dir}/ (one folder per benchmark+model).

    Files:
      - results.jsonl       per-question raw results (appended, tagged by mode)
      - detail.csv          per-question metrics with mode column (appended)
      - summary.xlsx        one sheet per mode, discipline x difficulty cross cells
      - skill_library.json  (search_skill mode only, written by eval function)

    If rt_writer is provided, JSONL and CSV were already written incrementally;
    only the XLSX summary is written.
    """
    import csv
    from openpyxl import Workbook, load_workbook

    os.makedirs(output_dir, exist_ok=True)

    # Tag each result with mode
    for r in results:
        r["mode"] = mode

    if rt_writer is None:
        # 1. JSONL: merge with existing (replace same mode, keep others)
        jsonl_path = os.path.join(output_dir, "results.jsonl")
        existing_lines = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("mode") != mode:
                            existing_lines.append(line)
                    except json.JSONDecodeError:
                        existing_lines.append(line)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for line in existing_lines:
                f.write(line + "\n")
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  [Saved] {jsonl_path} ({len(results)} lines, mode={mode})")

        # 2. CSV: merge with existing (replace same mode, keep others)
        csv_detail_path = os.path.join(output_dir, "detail.csv")
        detail_fields = ["mode", "id", "domain", "difficulty", "subfield", "correct", "score",
                         "total_tokens", "prompt_tokens", "completion_tokens",
                         "web_search_count", "num_rounds",
                         "explore_triggered", "num_queries", "num_page_visits",
                         "skills_generated", "skills_retrieved", "retrieved_skill_used",
                         "teacher_rewrite_count",
                         "ground_truth", "prediction"]
        existing_rows = []
        if os.path.exists(csv_detail_path):
            import csv as _csv
            with open(csv_detail_path, "r", encoding="utf-8", newline="") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    if row.get("mode") != mode:
                        existing_rows.append(row)
        with open(csv_detail_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)
            for r in results:
                writer.writerow(r)
        print(f"  [Saved] {csv_detail_path} ({len(results)} rows, mode={mode})")
    else:
        print(f"  [Saved] JSONL/CSV already written incrementally (mode={mode})")

    # 3. XLSX: one sheet per mode, discipline x difficulty cross cells only
    xlsx_path = os.path.join(output_dir, "summary.xlsx")

    agg_headers = ["discipline", "difficulty", "n", "correct", "accuracy",
                   "avg_tokens", "avg_prompt_tokens", "avg_completion_tokens",
                   "avg_web_searches", "avg_rounds"]

    agg_rows = []
    for domain in sorted(summary.get("by_discipline_x_difficulty", {}).keys()):
        for diff in sorted(summary["by_discipline_x_difficulty"][domain].keys()):
            s = summary["by_discipline_x_difficulty"][domain][diff]
            agg_rows.append([domain, diff, s["n"], s["correct"], s["acc"],
                             s["avg_tokens"], s["avg_prompt"], s["avg_completion"],
                             s["avg_searches"], s["avg_rounds"]])

    if os.path.exists(xlsx_path):
        wb = load_workbook(xlsx_path)
        if mode in wb.sheetnames:
            del wb[mode]
    else:
        wb = Workbook()
        if "Sheet" in wb.sheetnames:
            del wb["Sheet"]

    ws = wb.create_sheet(title=mode)
    ws.append(agg_headers)
    for row in agg_rows:
        ws.append(row)

    wb.save(xlsx_path)
    print(f"  [Saved] {xlsx_path} (sheet: {mode})")


# ---------------------------------------------------------------------------
# Summary TXT writer
# ---------------------------------------------------------------------------

def _write_summary_txt(output_dir: str, mode: str, summary: Dict, results: List[Dict]):
    """Write a human-readable summary_{mode}.txt to the output directory.

    Contents:
      - mode, accuracy
      - For free_tool / search_skill:
        explore_trigger_rate (questions with >=1 query AND >=1 page_visit)
      - For search_skill:
        skill_trigger_rate (retrieved skill actually used)
    """
    o = summary["overall"]
    txt_path = os.path.join(output_dir, f"summary_{mode}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"mode: {mode}\n")
        f.write(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"benchmark: {summary.get('benchmark', '?')}\n")
        f.write(f"model: {summary.get('model', '?')}\n")
        f.write(f"num_questions: {o['n']}\n")
        f.write(f"correct: {o['correct']}\n")
        f.write(f"accuracy: {o['acc']:.4f} ({o['acc']*100:.2f}%)\n")
        f.write(f"avg_tokens_per_question: {o['avg_tokens']:.1f}\n")
        f.write(f"avg_rounds_per_question: {o['avg_rounds']:.2f}\n")
        f.write(f"avg_web_searches_per_question: {o['avg_searches']:.2f}\n")

        # Explore trigger rate: questions where explore action was triggered
        if mode in ("free_tool", "search_skill"):
            explored_count = sum(
                1 for r in results
                if r.get("explore_triggered", False)
            )
            explore_trigger_rate = explored_count / len(results) if results else 0.0
            f.write(f"\n# Exploration stats (explore_triggered)\n")
            f.write(f"explore_trigger_count: {explored_count}/{len(results)}\n")
            f.write(f"explore_trigger_rate: {explore_trigger_rate:.4f} ({explore_trigger_rate*100:.2f}%)\n")

            # Average queries and page visits among explored
            explored_results = [r for r in results
                                if r.get("explore_triggered", False)]
            if explored_results:
                avg_q = sum(r.get("num_queries", 0) for r in explored_results) / len(explored_results)
                avg_pv = sum(r.get("num_page_visits", 0) for r in explored_results) / len(explored_results)
                f.write(f"avg_queries_per_explored: {avg_q:.2f}\n")
                f.write(f"avg_page_visits_per_explored: {avg_pv:.2f}\n")

        # Skill trigger rate (search_skill)
        if mode in ("search_skill",):
            has_retrieved = [r for r in results if r.get("skills_retrieved", 0) > 0]
            triggered = [r for r in has_retrieved if r.get("retrieved_skill_used")]
            skill_trigger_rate = len(triggered) / len(has_retrieved) if has_retrieved else 0.0
            f.write(f"\n# Skill stats\n")
            f.write(f"skill_trigger_count: {len(triggered)}/{len(has_retrieved)}\n")
            f.write(f"skill_trigger_rate: {skill_trigger_rate:.4f} ({skill_trigger_rate*100:.2f}%)\n")
            total_gen = sum(r.get("skills_generated", 0) for r in results)
            f.write(f"total_skills_generated: {total_gen}\n")

        # Per-domain breakdown
        f.write(f"\n# Per-domain accuracy\n")
        by_d = summary.get("by_discipline", {})
        for domain in sorted(by_d.keys()):
            s = by_d[domain]
            f.write(f"  {domain}: {s['correct']}/{s['n']} = {s['acc']*100:.2f}%\n")

    print(f"  [Saved] {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    args = parse_args()
    # Keep cascade_judge on the exact same config file as the main eval script.
    os.environ["EXSKILL_CONFIG_PATH"] = os.path.abspath(args.config)
    config = load_config(args.config)

    # Suppress agent per-round verbose output
    from core.agent import set_quiet
    set_quiet(True)

    # Resolve model profile
    model_profile = args.model or config["llm"].get("default")
    args.model = model_profile
    data_dir = config.get("benchmark", {}).get("data_dir", "data/")

    # Set judge profile for cascade_answer_judge (open-ended benchmarks)
    if args.judge_profile:
        os.environ["EXSKILL_REWARD_JUDGE_PRIMARY_PROFILE"] = args.judge_profile

    # Load benchmark
    benchmark = load_benchmark(args.benchmark)

    # List subsets
    if args.list_subsets:
        print(f"[{args.benchmark}] Loading dataset to list subsets...")
        load_path = args.data_file if args.data_file else data_dir
        items = benchmark.load(load_path, num_samples=-1, seed=args.seed)
        subsets = sorted(set(it.get("domain", "") for it in items))
        print(f"\nAvailable subsets ({len(subsets)}):")
        for s in subsets:
            count = sum(1 for it in items if it.get("domain") == s)
            print(f"  {s}: {count} questions")
        return

    # Print config
    print(f"[Config] Benchmark: {args.benchmark}")
    print(f"[Config] Modes: {args.mode}")
    print(f"[Config] Model: {model_profile}")
    print(f"[Config] Num questions: {args.num}")
    print(f"[Config] Subset: {args.subset or 'all'}")
    print(f"[Config] Concurrency: {args.concurrency}")
    print(f"[Config] Max rounds: {args.max_rounds}")
    print(f"[Config] Seed: {args.seed}")

    # Print model detail from profile
    profile_cfg = config.get("llm", {}).get("profiles", {}).get(model_profile, {})
    if profile_cfg:
        model_path = profile_cfg.get("model_path", "")
        enable_thinking = profile_cfg.get("enable_thinking", None)
        max_tokens = profile_cfg.get("max_tokens", "")
        port = profile_cfg.get("port", "")
        print(f"[Model] profile={model_profile}")
        if model_path:
            print(f"[Model]   model_path: {model_path}")
        if enable_thinking is not None:
            print(f"[Model]   enable_thinking: {enable_thinking}")
        if max_tokens:
            print(f"[Model]   max_tokens: {max_tokens}")
        if port:
            print(f"[Model]   port: {port}")

    # Load data
    if args.data_file:
        print(f"[Data] Loading from pre-extracted file: {args.data_file}")
        if args.data_file.endswith(".jsonl"):
            # JSONL files: delegate to benchmark.load() for proper field mapping
            items = benchmark.load(args.data_file, num_samples=args.num,
                                   subset=args.subset, seed=args.seed)
        else:
            with open(args.data_file, "r", encoding="utf-8") as f:
                items = json.load(f)
            if args.subset:
                items = [it for it in items if it.get("domain", "").lower() == args.subset.lower()]
            if args.num > 0 and args.num < len(items):
                import random as _random
                rng = _random.Random(args.seed)
                items = rng.sample(items, args.num)
        print(f"[Data] Loaded {len(items)} questions from file")
    else:
        print(f"[Data] Loading {args.benchmark}...")
        items = benchmark.load(data_dir, num_samples=args.num, subset=args.subset, seed=args.seed)
        print(f"[Data] Loaded {len(items)} questions")

    if args.shuffle_seed is not None:
        import random as _random
        rng = _random.Random(args.shuffle_seed)
        rng.shuffle(items)
        print(f"[Data] Shuffled questions with seed {args.shuffle_seed}")

    # Expand "full" → all modes, deduplicate while preserving order
    ALL_MODES = ["direct", "code", "free_tool", "search_skill", "search_skill_no_code", "skill_only", "skill_only_no_code"]
    expanded = []
    for m in args.mode:
        if m == "full":
            expanded.extend(ALL_MODES)
        else:
            expanded.append(m)
    # Deduplicate preserving order
    seen = set()
    modes = []
    for m in expanded:
        if m not in seen:
            seen.add(m)
            modes.append(m)
    args.mode = modes
    print(f"[Config] Expanded modes: {modes}")

    # Run evaluation
    t_start = time.time()
    output_base = args.output or "results/direct_eval"
    model_tag = (args.model or "default").replace("/", "_")
    _model_path = profile_cfg.get("model_path", "")
    if _model_path:
        import re
        _base = os.path.basename(_model_path)
        # Extract PID and step from model path like "..._20191_step80_hf"
        _m = re.search(r'(\d+)_step(\d+)', _base)
        if _m:
            model_tag = f"{model_tag}_{_m.group(1)}_step{_m.group(2)}"
        else:
            model_tag = f"{model_tag}_{_base}"
    if args.data_file:
        data_tag = os.path.splitext(os.path.basename(args.data_file))[0]
        output_dir = os.path.join(output_base, args.benchmark, f"{data_tag}_{model_tag}_{len(items)}")
    else:
        output_dir = os.path.join(output_base, args.benchmark, f"{model_tag}_{len(items)}")
    os.makedirs(output_dir, exist_ok=True)

    # Save run config to output dir
    run_config_path = os.path.join(output_dir, "config.txt")
    with open(run_config_path, "w", encoding="utf-8") as _cf:
        _cf.write(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        _cf.write(f"benchmark: {args.benchmark}\n")
        _cf.write(f"modes: {modes}\n")
        _cf.write(f"model_profile: {model_profile}\n")
        if profile_cfg:
            _cf.write(f"model_path: {profile_cfg.get('model_path', '')}\n")
            _cf.write(f"enable_thinking: {profile_cfg.get('enable_thinking', None)}\n")
            _cf.write(f"max_tokens: {profile_cfg.get('max_tokens', '')}\n")
            _cf.write(f"port: {profile_cfg.get('port', '')}\n")
            _cf.write(f"backend: {profile_cfg.get('backend', '')}\n")
        _cf.write(f"num_questions: {len(items)}\n")
        _cf.write(f"subset: {args.subset or 'all'}\n")
        _cf.write(f"concurrency: {args.concurrency}\n")
        _cf.write(f"max_rounds: {args.max_rounds}\n")
        _cf.write(f"seed: {args.seed}\n")
        _cf.write(f"shuffle_seed: {args.shuffle_seed if args.shuffle_seed is not None else 'original'}\n")
        _cf.write(f"data_file: {args.data_file or '(huggingface)'}\n")
        _cf.write(f"config_file: {args.config}\n")
        _cf.write(f"output_dir: {output_dir}\n")
    print(f"[Config] Saved to {run_config_path}")
    skill_cfg = config.get("skill", {})
    skill_top_k = args.skill_top_k or skill_cfg.get("skill_top_k", 3)
    retrieval_method = skill_cfg.get("retrieval", "embedding")
    # Prepare real-time writers for all modes upfront
    rt_writers = {}
    for m in modes:
        w = RealtimeWriter(output_dir, m)
        w.prepare(resume=args.resume)
        rt_writers[m] = w

    config["llm"]["default"] = model_profile
    has_web_mode = False
    num_runs = 3 if args.repeat_avg else 1
    repeat_run_summaries = {}

    run_schedule = []
    for m in modes:
        for ri in range(num_runs):
            run_schedule.append((m, ri))

    for mode_name, run_i in run_schedule:
        run_tag = f"{mode_name}_run{run_i+1}" if num_runs > 1 else mode_name

        print(f"\n{'='*60}")
        if num_runs > 1:
            print(f"  Starting mode: {mode_name}  (run {run_i+1}/{num_runs})")
        else:
            print(f"  Starting mode: {mode_name}")
        print(f"{'='*60}")

        if num_runs > 1:
            rt_writer = RealtimeWriter(output_dir, run_tag)
            rt_writer.prepare(resume=False)
            mode_items = items
        else:
            rt_writer = rt_writers[mode_name]
            # Resume: filter out already-completed items for this mode
            if args.resume:
                completed_ids = rt_writer.get_completed_ids()
                mode_items = [it for it in items if it["id"] not in completed_ids]
                if completed_ids:
                    print(f"  [Resume] Found {len(completed_ids)} completed, "
                          f"{len(mode_items)} remaining (of {len(items)} total)")
                if not mode_items:
                    print(f"  [Resume] All items already completed for mode '{mode_name}', skipping.")
                    continue
            else:
                mode_items = items

        if mode_name == "direct":
            llm = LLMClient.from_config(config, profile_name=model_profile)
            attach_usage_tracking(llm)
            results = await eval_direct(mode_items, benchmark, llm, args.concurrency,
                                        rt_writer=rt_writer)

        elif mode_name == "code":
            results = await eval_code(mode_items, benchmark, config, model_profile,
                                      args.concurrency, args.max_rounds,
                                      rt_writer=rt_writer)

        elif mode_name == "free_tool":
            has_web_mode = True
            results = await eval_free_tool(mode_items, benchmark, config, model_profile,
                                            args.concurrency, args.max_rounds,
                                            rt_writer=rt_writer)

        elif mode_name == "search_skill":
            has_web_mode = True
            skill_library = args.skill_library or os.path.join(
                output_dir, "skill_library.json")
            results = await eval_search_skill(mode_items, benchmark, config, model_profile,
                                              args.concurrency, args.max_rounds,
                                              skill_library, skill_top_k,
                                              retrieval=retrieval_method,
                                              batch_size=args.skill_batch_size,
                                              rt_writer=rt_writer,
                                              resume=args.resume)

        elif mode_name == "search_skill_no_code":
            has_web_mode = True
            skill_library = args.skill_library or os.path.join(
                output_dir, "skill_library.json")
            results = await eval_search_skill_no_code(mode_items, benchmark, config, model_profile,
                                              args.concurrency, args.max_rounds,
                                              skill_library, skill_top_k,
                                              retrieval=retrieval_method,
                                              batch_size=args.skill_batch_size,
                                              rt_writer=rt_writer,
                                              resume=args.resume)

        elif mode_name == "skill_only":
            skill_library = args.skill_library or os.path.join(
                output_dir, "skill_library.json")
            results = await eval_skill_only(mode_items, benchmark, config, model_profile,
                                            args.concurrency, args.max_rounds,
                                            skill_library, skill_top_k,
                                            rt_writer=rt_writer)

        elif mode_name == "skill_only_no_code":
            skill_library = args.skill_library or os.path.join(
                output_dir, "skill_library.json")
            results = await eval_skill_only_no_code(mode_items, benchmark, config, model_profile,
                                                     args.concurrency, args.max_rounds,
                                                     skill_library, skill_top_k,
                                                     rt_writer=rt_writer)

        # Save and print summary for this mode/run
        args.mode = run_tag
        if args.resume and num_runs == 1:
            all_results = []
            if os.path.exists(rt_writer.jsonl_path):
                with open(rt_writer.jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if obj.get("mode") == mode_name:
                                all_results.append(obj)
                        except json.JSONDecodeError:
                            continue
            print(f"  [Resume] Computing summary over {len(all_results)} total results "
                  f"({len(all_results) - len(results)} previous + {len(results)} new)")
            summary = compute_summary(all_results, args)
        else:
            all_results = results
            summary = compute_summary(results, args)

        if num_runs > 1:
            o = summary["overall"]
            print(f"  Run {run_i+1}/{num_runs}: "
                  f"{o['correct']}/{o['n']} = {_fmt_pct(o['acc'])}%")
            repeat_run_summaries.setdefault(mode_name, []).append(summary)
            save_results(results, summary, output_dir, run_tag, args,
                         rt_writer=rt_writer)
            _write_summary_txt(output_dir, run_tag, summary, all_results)
        else:
            print_summary(summary)
            save_results(results, summary, output_dir, mode_name, args,
                         rt_writer=rt_writer)
            _write_summary_txt(output_dir, mode_name, summary, all_results)

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

        avg_txt_path = os.path.join(output_dir, "repeat_avg_summary.txt")
        with open(avg_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Repeat average summary ({num_runs} runs)\n")
            f.write(f"{'='*60}\n")
            for line in avg_summary_lines:
                f.write(line + "\n")
        print(f"  [Saved] {avg_txt_path}")

    # Print page visit health check
    if has_web_mode:
        from core.agent import get_page_visit_stats
        pv = get_page_visit_stats()
        if pv["total"] > 0:
            effective = pv["success"] + pv["failed"]
            rate = pv["success"] / effective * 100 if effective > 0 else 0.0
            print(f"\n[PageVisit] total={pv['total']}  "
                  f"success={pv['success']}  failed={pv['failed']}  "
                  f"forbidden={pv['forbidden']}  "
                  f"success_rate={rate:.1f}%")
        else:
            print(f"\n[PageVisit] No page visits recorded")

    print(f"\n{'='*60}")
    print(f"  All modes completed: {modes}")
    print(f"  Total wall time: {time.time() - t_start:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
