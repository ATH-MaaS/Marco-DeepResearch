"""实验一：验证模型是否知道自己的知识缺口 (Knowledge Gap Self-Awareness)

流程：
  Step 1 (self_assess): 小模型面对问题，输出一个核心知识缺口（不作答）
  Step 2 (direct_answer): 小模型直接回答，用judge判断对错
  Step 3 (teacher_diagnose): 对做错的题，大模型诊断一个核心失败原因
  Step 4 (match): 大模型判断 Step1 自评 vs Step3 诊断是否匹配 (Yes/No)
  Step 5 (knowledge_verify): 对所有EXPLORE项，让学生解释gap内容，
          教师判断学生是否真的不知道 (LACKS/KNOWS)

核心指标：
  - Gap Match Rate: 做错题中，自评缺口与真实缺口匹配的比例
  - No-Gap Accuracy: 模型说"没有缺口"时的正确率
  - Gap Confirmed Rate: 探索项中，学生确实不知道gap内容的比例

用法：
  # 全量：500题，5个学生模型从小到大，教师固定 qwen37-max
  python scripts/exp1_knowledge_gap.py
  # 50题快速跑
  python scripts/exp1_knowledge_gap.py --num 50
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from llm.client import LLMClient
from core.decision import extract_answer_from_tag, _extract_tag_content_from_right

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SELF_ASSESS_PROMPT = """\
You are an intelligent assistant solving a problem.

## Question:
{question}

Rules:
- Think step-by-step in plain text, then output EXACTLY ONE action tag at the end.
- CRITICAL: Do NOT guess when uncertain. If you are not highly confident (>90%) in your answer, you MUST use <EXPLORE> to search for evidence first. Getting the answer wrong is worse than spending an extra round exploring.
- DEFAULT TO EXPLORE: For any problem involving mathematical theorems, specialized techniques, non-trivial algorithms, or domain-specific formulas, you should use <EXPLORE> as your FIRST action — even if you think you might know the answer. Exploration builds reusable skills and catches subtle errors that direct solving misses. Only skip exploration for truly elementary problems (basic arithmetic, simple algebra).

Uncertainty signals (any of these means you MUST EXPLORE):
- The question asks about specific facts, names, dates, numbers, or formulas you are not 100% sure about
- You do not know or are not fully sure about a concept, knowledge point, procedure, workflow, API, library usage, command, syntax, or implementation detail needed to solve the task
- You find yourself torn between two or more options
- You are relying on vague memory rather than solid knowledge
- The problem requires a specific algorithm, technique, or mathematical method that you are not fully confident you can implement correctly (e.g., a DP variant, number-theoretic formula, geometric algorithm, or specialized data structure)

Actions (Choose exactly one):
1) Explore — USE THIS when uncertain or missing knowledge: <EXPLORE>what specific information, knowledge, method, example, workflow, or implementation detail you need to verify</EXPLORE>
2) Answer — ONLY when confident: <ANSWER>final answer (or option letter like B for multiple choice)</ANSWER>"""

DIRECT_ANSWER_PROMPT = """\
Please answer the following question. Think step by step, then put your final answer inside <ANSWER></ANSWER> tags.

For multiple-choice questions, put only the option letter inside the tags, e.g. <ANSWER>B</ANSWER>.
For open-ended questions, put a concise answer inside the tags.

{question}"""

FINAL_ANSWER_PROMPT = "Based on the current context, please output your final answer in the following tag:\n<ANSWER>your final answer</ANSWER>"

TEACHER_DIAGNOSE_PROMPT = """\
A student model attempted to solve the following problem but got it wrong.

## Problem
{question}

## Ground Truth Answer
{ground_truth}

## Correct Solution
{solution}

## Student's Attempted Answer
{student_answer}

## Instructions
Identify the single most critical knowledge gap or reasoning error that caused the student to fail. Be specific — name the exact theorem, concept, formula, or technique that was missing or misapplied.

Output in this exact format:
<CORE_DIAGNOSIS>
[One sentence: the single most critical knowledge/capability that was missing or wrong]
</CORE_DIAGNOSIS>"""

KNOWLEDGE_VERIFY_PROMPT = """\
A student model claimed it needed to explore certain knowledge BEFORE attempting a problem. It then attempted to solve the problem directly (without exploration). Below is the student's full reasoning trace from that direct attempt.

## Problem
{question}

## Knowledge the Student Claimed to Need (self-assessed gap)
{core_gap}

## Student's Full Reasoning Trace (direct attempt without exploration)
{reasoning_trace}

## Student's Final Answer
{student_answer}

## Ground Truth Answer
{ground_truth}

## Instructions
Based on the student's ACTUAL REASONING TRACE, judge whether the student genuinely lacked the knowledge it claimed to need.

判定标准:
- KNOWS: The student ultimately applied the claimed knowledge correctly in its reasoning.
- LACKS: The student made substantive errors directly related to the claimed knowledge (wrong formula, incorrect concept, factual mistakes), OR completely failed to apply the knowledge and had to guess, OR avoided the topic entirely and used a workaround.

First output a brief reason (1-2 sentences) citing the specific evidence, then the verdict.

Format:
<REASON>your brief explanation</REASON>
<VERIFY>KNOWS</VERIFY> or <VERIFY>LACKS</VERIFY>"""

GAP_MATCH_PROMPT = """\
A model identified a knowledge gap BEFORE attempting a problem. A teacher later diagnosed the actual cause of failure AFTER the model got it wrong.

## Problem (for context)
{question}

## Model's Self-Assessed Gap (before attempting)
{self_gap}

## Teacher's Diagnosed Cause of Failure (after failure)
{teacher_gap}

## Instructions
Judge whether exploring the model's self-assessed gap would have plausibly helped avoid the diagnosed failure. The self-assessment may be broader or use different words — that's fine. The question is: if the model had searched for and learned about its self-assessed gap, would that knowledge have likely covered or prevented the specific error the teacher identified?

Output ONLY:
<MATCH>YES</MATCH> or <MATCH>NO</MATCH>"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def extract_tag(text: str, tag: str) -> str:
    content = _extract_tag_content_from_right(text, tag)
    return content.strip() if content is not None else ""


def extract_match(text: str) -> bool:
    val = extract_tag(text, "MATCH").strip().upper()
    return val == "YES"


def extract_verify(text: str) -> str:
    val = extract_tag(text, "VERIFY").strip().upper()
    if val in ("KNOWS", "LACKS"):
        return val
    return "UNKNOWN"


def extract_judgment(text: str) -> tuple:
    """Returns (judgment, core_gap), following parse_execution_step logic.
    EXPLORE checked before ANSWER (same priority as core/decision.py).
    No tag found → default CAN_SOLVE (model solved directly without tags).
    """
    explore = _extract_tag_content_from_right(text, "EXPLORE")
    if explore is not None:
        return "NEED_EXPLORE", explore.strip()
    answer = _extract_tag_content_from_right(text, "ANSWER")
    if answer is not None:
        return "CAN_SOLVE", answer.strip()
    return "CAN_SOLVE", ""


def load_items(data_path: str, num: int, seed: int,
               diff_min: float = 0.0, diff_max: float = 99.0) -> list:
    """Load items from local JSON/JSONL or HuggingFace.

    data_path can be:
      - Local .json file  (pre-extracted, like data/omnimath_hard_500.json)
      - Local .jsonl file (raw OmniMath JSONL)
      - "huggingface" / "hf" / "omnimath" → KbsdJames/Omni-MATH
      - "gpqa" / "gpqa_diamond" → Idavidrein/gpqa (Diamond, 198 items)
      - "gpqa_main" → Idavidrein/gpqa (Main split)
    """
    import random

    if data_path.lower() in ("huggingface", "hf", "omnimath"):
        data = _load_from_huggingface()
    elif data_path.lower() in ("gpqa", "gpqa_diamond", "gpqa-diamond"):
        data = _load_from_gpqa("gpqa_diamond")
    elif data_path.lower() in ("gpqa_main", "gpqa-main"):
        data = _load_from_gpqa("gpqa_main")
    elif data_path.endswith(".jsonl"):
        data = _load_from_jsonl(data_path)
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    def _parse_difficulty(x):
        try:
            return float(x.get("metadata", {}).get("difficulty", 0))
        except (ValueError, TypeError):
            return 0.0

    data = [x for x in data
            if diff_min <= _parse_difficulty(x) <= diff_max]
    rng = random.Random(seed)
    rng.shuffle(data)
    if num > 0 and num < len(data):
        data = data[:num]
    return data


def _load_from_huggingface() -> list:
    from datasets import load_dataset
    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    items = []
    for row in ds:
        domain_list = row.get("domain", [])
        domain = domain_list[0] if domain_list else "Mathematics"
        domain_short = domain.split("->")[1].strip() if "->" in domain else domain
        diff_val = row.get("difficulty", 0)
        items.append({
            "id": str(row.get("_idx", len(items))),
            "question": row["problem"],
            "options": [],
            "context": "",
            "ground_truth": row.get("answer", ""),
            "solution": row.get("solution", ""),
            "domain": domain_short,
            "metadata": {
                "subfield": domain,
                "difficulty": str(diff_val),
                "source": row.get("source", ""),
            },
        })
    return items


def _load_from_gpqa(config_name: str = "gpqa_diamond") -> list:
    """Load GPQA from HuggingFace (Idavidrein/gpqa).

    Each row has: Question, Correct Answer, Incorrect Answer 1/2/3, Explanation,
    Subdomain, High-level domain. We shuffle the 4 options deterministically per
    row so the correct letter is roughly uniform across A/B/C/D but reproducible.
    The shuffled prompt (with options) is baked into `question`; ground_truth is
    just the correct option letter.
    """
    import random as _random
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", config_name, split="train")
    items = []
    for idx, row in enumerate(ds):
        correct = (row.get("Correct Answer") or "").strip()
        incorrects = [
            (row.get("Incorrect Answer 1") or "").strip(),
            (row.get("Incorrect Answer 2") or "").strip(),
            (row.get("Incorrect Answer 3") or "").strip(),
        ]
        options = [correct] + incorrects
        rng = _random.Random(f"gpqa-{idx}")
        rng.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = "ABCD"[correct_idx]

        q_text = (row.get("Question") or "").strip()
        opt_block = "\n".join(f"{ch}. {opt}" for ch, opt in zip("ABCD", options))
        formatted_question = (
            f"{q_text}\n\nChoose the single best option. "
            f"Answer with only the option letter (A, B, C, or D).\n\n{opt_block}"
        )

        domain = (row.get("High-level domain") or "Science").strip()
        subdomain = (row.get("Subdomain") or domain).strip()

        items.append({
            "id": f"gpqa-{idx}",
            "question": formatted_question,
            "options": options,
            "context": "",
            "ground_truth": correct_letter,
            "solution": (row.get("Explanation") or "").strip(),
            "domain": domain,
            "metadata": {
                "subfield": subdomain,
                "difficulty": "8.0",
                "source": "GPQA",
                "correct_answer_text": correct,
                "config": config_name,
            },
        })
    return items


def _load_from_jsonl(path: str) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            domain_list = raw.get("domain", [])
            if isinstance(domain_list, list):
                domain = domain_list[0] if domain_list else "Science"
            else:
                domain = str(domain_list)
            domain_short = domain.split("->")[1].strip() if "->" in domain else domain
            question = raw.get("question") or raw.get("problem", "")
            ground_truth = raw.get("ground_truth") or raw.get("answer", "")
            qid = str(raw.get("id", raw.get("_idx", len(items))))
            diff_val = raw.get("metadata", {}).get("difficulty",
                         raw.get("difficulty", 0))
            items.append({
                "id": qid,
                "question": question,
                "options": raw.get("options", []),
                "context": raw.get("context", ""),
                "ground_truth": ground_truth,
                "solution": raw.get("solution", ""),
                "domain": domain_short,
                "subfield": raw.get("subfield", domain_short),
                "metadata": raw.get("metadata", {
                    "subfield": domain,
                    "difficulty": str(diff_val),
                    "source": raw.get("source", ""),
                }),
            })
    return items


# ---------------------------------------------------------------------------
# Judge (reuse cascade_answer_judge)
# ---------------------------------------------------------------------------

def judge_answer(question: str, prediction: str, ground_truth: str) -> dict:
    pred = prediction.strip()
    gt = ground_truth.strip()
    if len(gt) == 1 and gt.upper() in "ABCDEFGHIJ":
        if len(pred) <= 2 and pred and pred[0].upper() in "ABCDEFGHIJ":
            pred_letter = pred[0].upper()
            correct = pred_letter == gt.upper()
            return {"correct": correct, "score": 1.0 if correct else 0.0,
                    "detail": f"letter_match: pred={pred_letter}, gt={gt}"}
    from evaluation.cascade_judge import cascade_answer_judge
    score, detail = cascade_answer_judge(
        question=question,
        ground_truth=gt,
        prediction=pred,
    )
    return {"correct": score >= 0.5, "score": score, "detail": detail}


async def async_judge(question, prediction, ground_truth):
    import functools
    loop = asyncio.get_running_loop()
    fn = functools.partial(judge_answer, question, prediction, ground_truth)
    return await loop.run_in_executor(None, fn)


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

async def run_experiment(items: list, student_llm: LLMClient,
                         teacher_llm: LLMClient, concurrency: int,
                         output_dir: str, model_label: str = ""):
    sem = asyncio.Semaphore(concurrency)
    jsonl_path = os.path.join(output_dir, "results.jsonl")
    tag = f"[{model_label}]" if model_label else ""

    # ---- Resume: load already-completed records ----
    results = []
    done_ids = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = str(rec.get("id"))
                if qid in done_ids:
                    continue
                done_ids.add(qid)
                results.append(rec)
        if results:
            print(f"  {tag} resume: loaded {len(results)} done items from {jsonl_path}")

    pending_items = [it for it in items if str(it["id"]) not in done_ids]
    total = len(items)
    done_count = len(results)
    correct_count = sum(1 for r in results if r.get("step2_correct"))
    wrong_count = done_count - correct_count
    diagnosed_count = sum(1 for r in results
                          if not r.get("step2_correct") and not r.get("step1_no_gap"))
    match_count = sum(1 for r in results
                      if not r.get("step2_correct") and not r.get("step1_no_gap")
                      and r.get("step4_match") is True)

    async def process_one(item):
        nonlocal done_count, correct_count, wrong_count, match_count, diagnosed_count
        question = item["question"]
        options = item.get("options", [])
        if options:
            letters = "ABCDEFGHIJ"
            opt_lines = []
            for i, opt in enumerate(options):
                if opt.startswith(f"{letters[i]}. "):
                    opt_lines.append(opt)
                else:
                    opt_lines.append(f"{letters[i]}. {opt}")
            question += "\n\n" + "\n".join(opt_lines)
        ground_truth = item["ground_truth"]
        solution = item.get("solution", "")
        qid = item["id"]
        domain = item.get("domain", "")
        difficulty = item.get("metadata", {}).get("difficulty", "")

        result = {
            "id": qid,
            "domain": domain,
            "difficulty": difficulty,
            "question": question,
            "ground_truth": ground_truth,
        }

        async with sem:
            # --- Step 1: Self-Assessment (ANSWER or EXPLORE) ---
            t0 = time.time()
            self_assess_msg = [{"role": "user",
                                "content": SELF_ASSESS_PROMPT.format(question=question)}]
            self_assess_resp = await student_llm.chat(self_assess_msg)
            self_assess_resp = self_assess_resp or ""
            result["step1_response"] = self_assess_resp
            judgment, core_gap = extract_judgment(self_assess_resp)
            result["step1_judgment"] = judgment
            result["step1_core_gap"] = core_gap
            result["step1_no_gap"] = judgment == "CAN_SOLVE"
            result["step1_time"] = round(time.time() - t0, 2)

            if judgment == "CAN_SOLVE":
                # Model chose ANSWER directly — use that answer
                prediction = extract_answer_from_tag(self_assess_resp)
                if prediction:
                    result["step2_response_full"] = self_assess_resp
                    result["step2_prediction"] = prediction
                    result["step2_time"] = result["step1_time"]
                else:
                    # Model didn't output <ANSWER> tag — run direct solve
                    t0 = time.time()
                    answer_msgs = [{"role": "user",
                                    "content": DIRECT_ANSWER_PROMPT.format(question=question)}]
                    answer_resp = await student_llm.chat(answer_msgs, temperature=0.3)
                    answer_resp = answer_resp or ""
                    step2_full = answer_resp
                    prediction = extract_answer_from_tag(answer_resp)
                    if not prediction:
                        answer_msgs.append({"role": "assistant", "content": answer_resp})
                        answer_msgs.append({"role": "user", "content": FINAL_ANSWER_PROMPT})
                        followup = await student_llm.chat(answer_msgs, temperature=0.3)
                        followup = followup or ""
                        step2_full = answer_resp + "\n\n[FOLLOWUP]\n" + followup
                        prediction = extract_answer_from_tag(followup)
                        if not prediction:
                            prediction = answer_resp.strip()
                    result["step2_response_full"] = step2_full
                    result["step2_prediction"] = prediction
                    result["step2_time"] = round(time.time() - t0, 2)
            else:
                # Model chose EXPLORE — run a separate direct solve
                t0 = time.time()
                answer_msgs = [{"role": "user",
                                "content": DIRECT_ANSWER_PROMPT.format(question=question)}]
                answer_resp = await student_llm.chat(answer_msgs, temperature=0.3)
                answer_resp = answer_resp or ""
                step2_full = answer_resp

                prediction = extract_answer_from_tag(answer_resp)
                if not prediction:
                    answer_msgs.append({"role": "assistant", "content": answer_resp})
                    answer_msgs.append({"role": "user", "content": FINAL_ANSWER_PROMPT})
                    followup = await student_llm.chat(answer_msgs, temperature=0.3)
                    followup = followup or ""
                    step2_full = answer_resp + "\n\n[FOLLOWUP]\n" + followup
                    prediction = extract_answer_from_tag(followup)
                    if not prediction:
                        prediction = answer_resp.strip()

                result["step2_response_full"] = step2_full
                result["step2_prediction"] = prediction
                result["step2_time"] = round(time.time() - t0, 2)

        # Judge
        eval_result = await async_judge(question, prediction, ground_truth)
        result["step2_correct"] = eval_result["correct"]
        result["step2_score"] = eval_result["score"]
        result["step2_judge_detail"] = eval_result.get("detail", "")

        if result["step2_correct"]:
            sym = "+"
        else:
            sym = "-"

        # --- Step 3 & 4: Only for wrong answers ---
        result["step3_core_diagnosis"] = ""
        result["step3_response_raw"] = ""
        result["step4_match"] = None
        result["step4_response_raw"] = ""
        if not result["step2_correct"]:
            async with sem:
                # Step 3: Teacher diagnoses one core gap
                t0 = time.time()
                diag_msgs = [{"role": "user",
                              "content": TEACHER_DIAGNOSE_PROMPT.format(
                                  question=question,
                                  ground_truth=ground_truth,
                                  solution=solution,
                                  student_answer=prediction)}]
                diag_resp = await teacher_llm.chat(diag_msgs)
                diag_resp = diag_resp or ""
                result["step3_response_raw"] = diag_resp
                result["step3_core_diagnosis"] = extract_tag(diag_resp, "CORE_DIAGNOSIS")
                result["step3_time"] = round(time.time() - t0, 2)

                # Step 4: Match self-assessed gap vs diagnosed gap (Yes/No)
                if result["step1_no_gap"]:
                    result["step4_match"] = False
                else:
                    t0 = time.time()
                    match_msgs = [{"role": "user",
                                   "content": GAP_MATCH_PROMPT.format(
                                       question=question,
                                       self_gap=result["step1_core_gap"],
                                       teacher_gap=result["step3_core_diagnosis"])}]
                    match_resp = await teacher_llm.chat(match_msgs, temperature=0.0)
                    match_resp = match_resp or ""
                    result["step4_response_raw"] = match_resp
                    result["step4_match"] = extract_match(match_resp)
                    result["step4_time"] = round(time.time() - t0, 2)

        # --- Step 5: Knowledge Verification (for all EXPLORE items) ---
        # Use the Step 2 reasoning trace as behavioral evidence instead of
        # asking the student to self-report.  The teacher checks whether the
        # trace shows the student actually possessed the claimed knowledge.
        result["step5_verify"] = None
        result["step5_judge_response"] = ""
        if not result["step1_no_gap"] and result["step1_core_gap"]:
            async with sem:
                t0 = time.time()
                reasoning_trace = result.get("step2_response_full", "")
                verify_msgs = [{"role": "user",
                                "content": KNOWLEDGE_VERIFY_PROMPT.format(
                                    question=question,
                                    core_gap=result["step1_core_gap"],
                                    reasoning_trace=reasoning_trace,
                                    student_answer=prediction,
                                    ground_truth=ground_truth)}]
                verify_resp = await teacher_llm.chat(verify_msgs, temperature=0.0)
                verify_resp = verify_resp or ""
                result["step5_judge_response"] = verify_resp
                result["step5_verify"] = extract_verify(verify_resp)
                result["step5_reason"] = extract_tag(verify_resp, "REASON")
                result["step5_time"] = round(time.time() - t0, 2)

        # ---- Atomic counter update + progress snapshot ----
        done_count += 1
        if result["step2_correct"]:
            correct_count += 1
        else:
            wrong_count += 1
            diagnosed_count += 1
            if result["step4_match"]:
                match_count += 1

        snap_done = done_count
        snap_correct = correct_count
        snap_diagnosed = diagnosed_count
        snap_match = match_count

        acc = snap_correct / snap_done * 100
        match_info = ""
        if snap_diagnosed > 0:
            match_rate = snap_match / snap_diagnosed * 100
            match_info = f"  match_rate={match_rate:.0f}%({snap_match}/{snap_diagnosed})"
        gap_info = ""
        if result["step1_judgment"] == "NEED_EXPLORE" and result["step1_core_gap"]:
            verify_tag = ""
            if result["step5_verify"]:
                verify_tag = f" v={result['step5_verify']}"
            gap_info = f"  gap=[{result['step1_core_gap'][:80]}]{verify_tag}"
        print(f"  {tag}[{snap_done}/{total}] {sym} id={qid} "
              f"j={result['step1_judgment'][:4]} "
              f"acc={acc:.1f}%{match_info}{gap_info}")

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return result

    if not pending_items:
        print(f"  {tag} all {total} items already done, skipping.")
        return results

    tasks = [asyncio.create_task(process_one(item)) for item in pending_items]
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(results: list, output_dir: str, args=None):
    total = len(results)
    correct_items = [r for r in results if r["step2_correct"]]
    wrong_items = [r for r in results if not r["step2_correct"]]
    correct = len(correct_items)
    wrong = len(wrong_items)

    correct_explore = sum(1 for r in correct_items if not r["step1_no_gap"])
    correct_explore_rate = correct_explore / correct * 100 if correct else 0

    wrong_explore = sum(1 for r in wrong_items if not r["step1_no_gap"])
    wrong_explore_rate = wrong_explore / wrong * 100 if wrong else 0

    explore_wrong_items = [r for r in wrong_items if not r["step1_no_gap"]]
    matched = sum(1 for r in explore_wrong_items if r.get("step4_match") is True)
    match_total = len(explore_wrong_items)
    match_rate = matched / match_total * 100 if match_total else 0

    explore_items = [r for r in results if not r["step1_no_gap"]]
    verified_total = sum(1 for r in explore_items if r.get("step5_verify") in ("KNOWS", "LACKS"))
    verified_lacks = sum(1 for r in explore_items if r.get("step5_verify") == "LACKS")
    verified_knows = sum(1 for r in explore_items if r.get("step5_verify") == "KNOWS")
    gap_confirmed_rate = verified_lacks / verified_total * 100 if verified_total else 0

    model_name = args.student if args else "model"

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1: Knowledge Gap Self-Awareness")
    print(f"{'='*80}")
    print(f"Model: {model_name}  |  Total: {total}  "
          f"Correct: {correct} ({correct/total*100:.1f}%)  Wrong: {wrong}")

    print(f"\n{'─'*80}")
    print(f"  {'Model':<15} {'Acc':>6} {'做对探索率':>12} {'做错探索率':>12} {'Gap Match':>12} {'Gap Confirmed':>14}")
    print(f"  {'─'*68}")
    print(f"  {model_name:<15} {correct/total*100:>5.1f}% "
          f"{correct_explore_rate:>10.1f}%  "
          f"{wrong_explore_rate:>10.1f}%  "
          f"{match_rate:>10.1f}%  "
          f"{gap_confirmed_rate:>12.1f}%")
    print(f"{'─'*80}")

    print(f"\n  做对探索率: 做对的 {correct} 题中，{correct_explore} 题触发了探索 ({correct_explore_rate:.1f}%)")
    print(f"  做错探索率: 做错的 {wrong} 题中，{wrong_explore} 题触发了探索 ({wrong_explore_rate:.1f}%)")
    print(f"  Gap Match:  探索+做错的 {match_total} 题中，{matched} 题关键词匹配 ({match_rate:.1f}%)")
    print(f"  Gap Confirmed: 所有探索 {verified_total} 题中，{verified_lacks} 题确认不知道 ({gap_confirmed_rate:.1f}%)，"
          f"{verified_knows} 题其实已知")

    correct_explore_items = [r for r in correct_items if not r["step1_no_gap"]]
    wrong_explore_items = [r for r in wrong_items if not r["step1_no_gap"]]
    ce_verified = sum(1 for r in correct_explore_items if r.get("step5_verify") in ("KNOWS", "LACKS"))
    ce_lacks = sum(1 for r in correct_explore_items if r.get("step5_verify") == "LACKS")
    ce_knows = sum(1 for r in correct_explore_items if r.get("step5_verify") == "KNOWS")
    we_verified = sum(1 for r in wrong_explore_items if r.get("step5_verify") in ("KNOWS", "LACKS"))
    we_lacks = sum(1 for r in wrong_explore_items if r.get("step5_verify") == "LACKS")
    we_knows = sum(1 for r in wrong_explore_items if r.get("step5_verify") == "KNOWS")
    ce_rate_str = f"{ce_lacks/ce_verified*100:.1f}%" if ce_verified else "N/A"
    we_rate_str = f"{we_lacks/we_verified*100:.1f}%" if we_verified else "N/A"
    print(f"    答对+EXPLORE ({len(correct_explore_items)} items): "
          f"LACKS={ce_lacks} KNOWS={ce_knows} → Confirmed: {ce_rate_str}")
    print(f"    答错+EXPLORE ({len(wrong_explore_items)} items): "
          f"LACKS={we_lacks} KNOWS={we_knows} → Confirmed: {we_rate_str}")

    # Per-domain
    domains = sorted(set(r["domain"] for r in results))
    if len(domains) > 1:
        print(f"\n  {'Domain':<18} {'N':>4} {'Acc':>6} {'做对探索率':>10} {'做错探索率':>10} {'Match':>8}")
        print(f"  {'─'*60}")
        for dom in domains:
            dr = [r for r in results if r["domain"] == dom]
            dc = [r for r in dr if r["step2_correct"]]
            dw = [r for r in dr if not r["step2_correct"]]
            dc_exp = sum(1 for r in dc if not r["step1_no_gap"])
            dw_exp = sum(1 for r in dw if not r["step1_no_gap"])
            dw_exp_match = sum(1 for r in dw if not r["step1_no_gap"] and r.get("step4_match") is True)
            c_rate = f"{dc_exp/len(dc)*100:.0f}%" if dc else "N/A"
            w_rate = f"{dw_exp/len(dw)*100:.0f}%" if dw else "N/A"
            m_rate = f"{dw_exp_match}/{dw_exp}" if dw_exp else "N/A"
            print(f"  {dom:<18} {len(dr):>4} {len(dc)/len(dr)*100:>5.1f}% "
                  f"{c_rate:>10} {w_rate:>10} {m_rate:>8}")

    # Save
    summary = {
        "model": model_name,
        "total": total, "correct": correct, "wrong": wrong,
        "accuracy_pct": round(correct / total * 100, 2),
        "correct_explore_count": correct_explore,
        "correct_explore_rate_pct": round(correct_explore_rate, 2),
        "wrong_explore_count": wrong_explore,
        "wrong_explore_rate_pct": round(wrong_explore_rate, 2),
        "gap_match_count": matched,
        "gap_match_total": match_total,
        "gap_match_rate_pct": round(match_rate, 2),
        "gap_confirmed_total": verified_total,
        "gap_confirmed_lacks": verified_lacks,
        "gap_confirmed_knows": verified_knows,
        "gap_confirmed_rate_pct": round(gap_confirmed_rate, 2),
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Exp1: Knowledge Gap Self-Awareness")
    p.add_argument("--config", type=str, default="config/config_api.yaml")
    p.add_argument("--data_file", type=str, default="omnimath",
                   help="Dataset source: 'omnimath'/'huggingface' (default, KbsdJames/Omni-MATH), "
                        "'gpqa_diamond' (198 q), 'gpqa_main', or local .json/.jsonl")
    p.add_argument("--student", type=str, nargs="+",
                   default=["qwen3-8b-api", "qwen3-14b-api", "qwen3-32b-api",
                            "qwen3-235b-a22b-api", "qwen3-max-api"],
                   help="Student model profile name(s), small to large")
    p.add_argument("--teacher", type=str, default="qwen37-max",
                   help="Teacher model profile name (for diagnosis & consistency)")
    p.add_argument("--num", type=int, default=500,
                   help="Number of questions to use (OmniMath has 4428, "
                        "52d9 at difficulty >= 7.5)")
    p.add_argument("--diff_min", type=float, default=8.0,
                   help="Minimum difficulty (inclusive)")
    p.add_argument("--diff_max", type=float, default=9.5,
                   help="Maximum difficulty (inclusive)")
    p.add_argument("--concurrency", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def compute_metrics(results: list) -> dict:
    total = len(results)
    correct_items = [r for r in results if r["step2_correct"]]
    wrong_items = [r for r in results if not r["step2_correct"]]
    correct = len(correct_items)
    wrong = len(wrong_items)
    correct_explore = sum(1 for r in correct_items if not r["step1_no_gap"])
    wrong_explore = sum(1 for r in wrong_items if not r["step1_no_gap"])
    explore_wrong_items = [r for r in wrong_items if not r["step1_no_gap"]]
    matched = sum(1 for r in explore_wrong_items if r.get("step4_match") is True)
    match_total = len(explore_wrong_items)

    explore_items = [r for r in results if not r["step1_no_gap"]]
    verified_total = sum(1 for r in explore_items if r.get("step5_verify") in ("KNOWS", "LACKS"))
    verified_lacks = sum(1 for r in explore_items if r.get("step5_verify") == "LACKS")
    verified_knows = sum(1 for r in explore_items if r.get("step5_verify") == "KNOWS")

    correct_explore_items = [r for r in correct_items if not r["step1_no_gap"]]
    wrong_explore_items = [r for r in wrong_items if not r["step1_no_gap"]]
    ce_verified = sum(1 for r in correct_explore_items if r.get("step5_verify") in ("KNOWS", "LACKS"))
    ce_lacks = sum(1 for r in correct_explore_items if r.get("step5_verify") == "LACKS")
    ce_knows = sum(1 for r in correct_explore_items if r.get("step5_verify") == "KNOWS")
    we_verified = sum(1 for r in wrong_explore_items if r.get("step5_verify") in ("KNOWS", "LACKS"))
    we_lacks = sum(1 for r in wrong_explore_items if r.get("step5_verify") == "LACKS")
    we_knows = sum(1 for r in wrong_explore_items if r.get("step5_verify") == "KNOWS")

    return {
        "total": total, "correct": correct, "wrong": wrong,
        "accuracy_pct": round(correct / total * 100, 2) if total else 0,
        "correct_explore": correct_explore,
        "correct_explore_rate": round(correct_explore / correct * 100, 2) if correct else 0,
        "wrong_explore": wrong_explore,
        "wrong_explore_rate": round(wrong_explore / wrong * 100, 2) if wrong else 0,
        "gap_matched": matched,
        "gap_match_total": match_total,
        "gap_match_rate": round(matched / match_total * 100, 2) if match_total else 0,
        "verify_total": verified_total,
        "verify_lacks": verified_lacks,
        "verify_knows": verified_knows,
        "gap_confirmed_rate": round(verified_lacks / verified_total * 100, 2) if verified_total else 0,
        "ce_total": len(correct_explore_items),
        "ce_verified": ce_verified,
        "ce_lacks": ce_lacks,
        "ce_knows": ce_knows,
        "ce_confirmed_rate": round(ce_lacks / ce_verified * 100, 2) if ce_verified else 0,
        "we_total": len(wrong_explore_items),
        "we_verified": we_verified,
        "we_lacks": we_lacks,
        "we_knows": we_knows,
        "we_confirmed_rate": round(we_lacks / we_verified * 100, 2) if we_verified else 0,
    }


def print_combined_table(all_metrics: dict):
    """Print a combined table for all student models."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1: Knowledge Gap Self-Awareness — Combined Results")
    print(f"{'='*80}")
    print(f"\n  {'Model':<18} {'Acc':>6} {'做对探索率':>12} {'做错探索率':>12} {'Gap Match':>12} {'Gap Confirmed':>14}")
    print(f"  {'─'*72}")
    for model_name, m in all_metrics.items():
        print(f"  {model_name:<18} {m['accuracy_pct']:>5.1f}% "
              f"{m['correct_explore_rate']:>10.1f}%  "
              f"{m['wrong_explore_rate']:>10.1f}%  "
              f"{m['gap_match_rate']:>10.1f}%  "
              f"{m['gap_confirmed_rate']:>12.1f}%")
    print(f"  {'─'*72}")
    print()
    for model_name, m in all_metrics.items():
        print(f"  [{model_name}]")
        print(f"    做对探索率: 做对 {m['correct']} 题中 {m['correct_explore']} 题触发探索")
        print(f"    做错探索率: 做错 {m['wrong']} 题中 {m['wrong_explore']} 题触发探索")
        print(f"    Gap Match:  探索+做错 {m['gap_match_total']} 题中 {m['gap_matched']} 题匹配")
        print(f"    Gap Confirmed: 所有探索 {m['verify_total']} 题中 {m['verify_lacks']} 题确认不知道 "
              f"({m['gap_confirmed_rate']:.1f}%), {m['verify_knows']} 题其实已知")
        ce_rate = f"{m['ce_confirmed_rate']:.1f}%" if m['ce_verified'] else "N/A"
        we_rate = f"{m['we_confirmed_rate']:.1f}%" if m['we_verified'] else "N/A"
        print(f"      答对+EXPLORE ({m['ce_total']} items): "
              f"LACKS={m['ce_lacks']} KNOWS={m['ce_knows']} → Confirmed: {ce_rate}")
        print(f"      答错+EXPLORE ({m['we_total']} items): "
              f"LACKS={m['we_lacks']} KNOWS={m['we_knows']} → Confirmed: {we_rate}")


def main():
    args = parse_args()
    os.environ["EXSKILL_CONFIG_PATH"] = os.path.abspath(args.config)
    config = load_config(args.config)

    teacher_llm = LLMClient.from_config(config, profile_name=args.teacher)
    print(f"Teacher model: {args.teacher} ({teacher_llm.default_model})")
    print(f"Student models: {args.student}")

    items = load_items(args.data_file, args.num, args.seed,
                       diff_min=args.diff_min, diff_max=args.diff_max)
    print(f"Loaded {len(items)} questions (difficulty {args.diff_min}-{args.diff_max} from {args.data_file})")
    def _safe_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    diffs = [_safe_float(x.get("metadata", {}).get("difficulty", 0)) for x in items]
    if diffs:
        print(f"Difficulty range: {min(diffs):.1f} - {max(diffs):.1f}")
    else:
        print("No difficulty values found")

    base_output = args.output or os.path.join(
        "results", "exp1_knowledge_gap",
        f"{'_'.join(args.student)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(base_output, exist_ok=True)

    run_config = {
        "students": args.student,
        "teacher": args.teacher,
        "data_file": args.data_file,
        "num": args.num,
        "diff_min": args.diff_min,
        "diff_max": args.diff_max,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "timestamp": datetime.now().isoformat(),
    }

    # Prepare specs (one per student model). Each model has its own internal
    # Semaphore(concurrency) inside run_experiment, while all student models
    # run concurrently in the same event loop.
    student_specs = []
    for student_name in args.student:
        student_llm = LLMClient.from_config(config, profile_name=student_name)
        output_dir = os.path.join(base_output, student_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "run_config.json"), "w") as f:
            json.dump({**run_config, "current_student": student_name}, f, indent=2)
        print(f"  Prepared: {student_name} ({student_llm.default_model}) -> {output_dir}")
        student_specs.append((student_name, student_llm, output_dir))

    print(f"\nLaunching {len(student_specs)} students in parallel "
          f"(per-student concurrency={args.concurrency})\n")

    async def run_all():
        coros = [
            run_experiment(items, llm, teacher_llm, args.concurrency,
                           output_dir, model_label=name)
            for name, llm, output_dir in student_specs
        ]
        return await asyncio.gather(*coros)

    all_results = asyncio.run(run_all())

    all_metrics = {}
    for (student_name, _, output_dir), results in zip(student_specs, all_results):
        m = compute_metrics(results)
        m["model"] = student_name
        all_metrics[student_name] = m

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        print(f"  {student_name} done. Acc={m['accuracy_pct']}% "
              f"做对探索率={m['correct_explore_rate']}% "
              f"做错探索率={m['wrong_explore_rate']}% "
              f"GapMatch={m['gap_match_rate']}%")

    # Combined table
    print_combined_table(all_metrics)

    combined_path = os.path.join(base_output, "combined_summary.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"Combined summary saved to {combined_path}")


if __name__ == "__main__":
    main()
