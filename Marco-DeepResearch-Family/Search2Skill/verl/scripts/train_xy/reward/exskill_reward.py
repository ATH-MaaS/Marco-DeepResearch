"""Search2Skill reward function for GRPO training.

Implements the reward design from the paper:
    R(τ) = clip_{[0,1]}(λ_a · R_base(τ) + λ_c · s^c(τ))

where s^c(τ) gates the quality bonus q(τ) via contrastive choice,
and q(τ) = max(0, ½(s_reuse + s_ground) - p_q).
"""
import ast
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from evaluation.cascade_judge import (
    cascade_answer_judge,
    extract_final_answer,
    reward_profiles,
    sync_chat,
    sync_chat_messages,
)

from core.decision import parse_execution_step, extract_answer_from_tag, _parse_query_content
from core.skill_schema import skill_workflow, skill_code

# Single reward configuration (paper final version):
#   λ_a = 0.7, λ_c = 0.3, τ_0 = 0.25, ρ = 0.15
REWARD_CONFIG = {
    "use_contrastive_choice": True,
    "s_a": 0.7,
    "s_c": 0.3,
    "contrastive_threshold": 0.25,
    "use_flow_penalty": True,
    "use_skill_quality": True,
}

ACTION_TAGS = ("ANSWER", "ACTIVATE_SKILL", "CODE", "EXPLORE", "QUERY", "PAGE_VISIT", "GENERATE_SKILL")
THINK_NORMAL_LIMIT = 3000
THINK_WARNING_LIMIT = 6000
THINK_SEVERE_LIMIT = 12000

QUERY_PENALTY_RHO = float(os.environ.get("QUERY_PENALTY_RHO", "0.15"))


def _extract_assistant_text(solution_str: str, extra_info: dict[str, Any]) -> str:
    messages = extra_info.get("conversation_messages")
    if messages:
        return "\n".join(
            msg["content"]
            for msg in messages
            if msg.get("role") == "assistant" and msg.get("content")
        )
    return solution_str or ""


def parse_generate_skill(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    candidates = [raw]
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        candidates.insert(0, json_match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(obj, dict) and obj.get("skill_name"):
            return obj
    return None


_SMOKE_DONE = False


def compute_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    del kwargs
    import time as _time

    global _SMOKE_DONE
    if not _SMOKE_DONE:
        _get_reuse_eval_infra()
        _get_query_judge_infra()
        _SMOKE_DONE = True

    _t0 = _time.time()
    max_workers = int(os.environ.get("REWARD_PARALLEL_WORKERS", 32))
    args_list = list(zip(data_sources, solution_strs, ground_truths, extra_infos, strict=True))

    _reward_sample_timings: list[dict[str, float]] = []

    outputs = [None] * len(args_list)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_compute_single, ds, sol, gt, ei, _reward_sample_timings): idx
            for idx, (ds, sol, gt, ei) in enumerate(args_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            outputs[idx] = future.result()
    _elapsed = _time.time() - _t0
    n = len(args_list)
    _answer_times = [t["answer_judge"] for t in _reward_sample_timings if "answer_judge" in t]
    print(f"[Reward Timing] total={_elapsed:.1f}s n={n} workers={max_workers}", flush=True)
    if _answer_times:
        print(f"  answer_judge:  sum={sum(_answer_times):.1f}s  avg={sum(_answer_times)/len(_answer_times):.2f}s  max={max(_answer_times):.1f}s", flush=True)

    outputs = _apply_contrastive_choice(outputs, REWARD_CONFIG)

    return outputs


# ---------------------------------------------------------------------------
# Contrastive choice reward (when-to-explore gate)
# ---------------------------------------------------------------------------

def _apply_contrastive_choice(outputs: list[dict[str, Any]], version_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Group-level contrastive choice reward.

    Compares explore vs direct average correctness within each group.
    - explore significantly better -> explore gets s_c = q(τ), direct gets s_c = 0
    - direct significantly better  -> direct gets s_c = 1.0, explore gets s_c = 0
    - similar or no valid contrast -> all get s_c = 0 (no choice signal)
    """
    import time as _time

    rollout_n = int(os.environ.get("ROLLOUT_N", 16))
    s_c_weight = float(version_cfg.get("s_c", 0.3))
    threshold = float(os.environ.get("CONTRASTIVE_THRESHOLD", version_cfg.get("contrastive_threshold", 0.25)))
    min_count = int(os.environ.get("CONTRASTIVE_MIN_COUNT", 3))
    n = len(outputs)
    if n % rollout_n != 0:
        print(f"[contrastive_choice] WARN: batch size {n} not divisible by rollout_n {rollout_n}, skipping", flush=True)
        return outputs

    reuse_eval_count = 0
    reuse_eval_time = 0.0
    query_penalty_eval_count = 0
    query_penalty_eval_time = 0.0

    for g in range(n // rollout_n):
        group = outputs[g * rollout_n : (g + 1) * rollout_n]
        explore_indices = [i for i, o in enumerate(group) if o.get("used_explore", False)]
        direct_indices = [i for i, o in enumerate(group) if not o.get("used_explore", False)]

        if len(explore_indices) < min_count or len(direct_indices) < min_count:
            for i in range(len(group)):
                group[i]["contrastive_label"] = "no_contrast"
                group[i]["s_c"] = 0.0
                outputs[g * rollout_n + i] = group[i]
            continue

        explore_avg = sum(group[i]["answer_score"] for i in explore_indices) / len(explore_indices)
        direct_avg = sum(group[i]["answer_score"] for i in direct_indices) / len(direct_indices)
        diff = explore_avg - direct_avg

        if diff >= threshold:
            # Batch query penalty evaluation for explore samples (parallel)
            query_penalties: dict[int, tuple[float, dict[str, Any]]] = {}
            _t_qp_batch = _time.time()
            qp_deadline = _time.time() + float(os.environ.get("QUERY_JUDGE_TOTAL_TIMEOUT", "300"))
            qp_tasks = {}
            for i in explore_indices:
                o = group[i]
                q = o.get("_question", "")
                conv = o.get("_conversation_messages") or []
                if q and conv:
                    qp_tasks[i] = (q, conv)
            if qp_tasks:
                remaining = max(0, qp_deadline - _time.time())
                qp_workers = min(len(qp_tasks), int(os.environ.get("QUERY_JUDGE_WORKERS", "8")))
                with ThreadPoolExecutor(max_workers=qp_workers) as qp_pool:
                    future_to_i = {
                        qp_pool.submit(_compute_query_penalty_for_rollout, q, conv): i
                        for i, (q, conv) in qp_tasks.items()
                    }
                    try:
                        for future in as_completed(future_to_i, timeout=remaining if remaining > 0 else 1):
                            idx = future_to_i[future]
                            try:
                                query_penalties[idx] = future.result(timeout=0)
                                query_penalty_eval_count += 1
                            except Exception as e:
                                print(f"[query_penalty] group={g} idx={idx} failed: {e}", flush=True)
                                query_penalties[idx] = (0.0, {"reason": f"error:{e}"})
                    except TimeoutError:
                        n_missed = sum(1 for i in qp_tasks if i not in query_penalties)
                        print(f"[query_penalty] group={g} deadline exceeded, {n_missed}/{len(qp_tasks)} missed", flush=True)
            query_penalty_eval_time += _time.time() - _t_qp_batch

            for i in explore_indices:
                o = group[i]
                fmt = o.get("format_score", 0) or 0

                # s_reuse: skill reuse score
                gen_skills = o.get("_generated_skills") or []
                sim_qs = o.get("_similar_questions") or []
                if gen_skills and sim_qs:
                    _t_reuse = _time.time()
                    reuse_score, reuse_details = _evaluate_skill_reuse_sync(gen_skills, sim_qs)
                    reuse_eval_time += _time.time() - _t_reuse
                    reuse_eval_count += 1
                    o["skill_reuse_score"] = round(reuse_score, 4)
                    o["skill_reuse_raw_delta"] = round(reuse_details.get("raw_mean_delta", 0.0), 4)
                    o["k_similar_evaluated"] = int(reuse_details.get("k_evaluated", 0))
                    o["skill_reuse_reason"] = str(reuse_details.get("reason", ""))
                else:
                    reuse_score = 0.5
                    o["skill_reuse_score"] = 0.5
                    o["skill_reuse_reason"] = "no_skill_or_no_similar"

                # s_ground: grounding score from skill quality judge, normalized [1,10] -> [0,1]
                grounding_raw = o.get("skill_quality_grounding_raw", 5)
                s_ground = max(0.0, min(1.0, (grounding_raw - 1) / 9.0))

                # p_q: query penalty
                p_q = 0.0
                qp_details = None
                if i in query_penalties:
                    p_q_raw, qp_details = query_penalties[i]
                    p_q = abs(p_q_raw)
                    o["query_penalty"] = round(p_q, 4)
                    o["query_penalty_dim_scores"] = {
                        "query_abstraction": (qp_details or {}).get("query_abstraction_score"),
                        "evidence_gain": (qp_details or {}).get("evidence_gain_score"),
                    }
                else:
                    o["query_penalty"] = 0.0
                    o["query_penalty_dim_scores"] = {}

                # q(τ) = max(0, ½(s_reuse + s_ground) - p_q)
                quality_bonus = max(0.0, 0.5 * (reuse_score + s_ground) - p_q)
                s_c = quality_bonus

                o["s_c"] = round(s_c, 4)
                o["s_ground"] = round(s_ground, 4)
                o["score"] = round(max(0.0, min(1.0, o["score"] + s_c_weight * s_c)), 4)
                o["contrastive_label"] = "explore_rewarded"
                outputs[g * rollout_n + i] = o

            for i in direct_indices:
                o = group[i]
                o["s_c"] = 0.0
                o["contrastive_label"] = "direct_punished"
                outputs[g * rollout_n + i] = o

            n_query_penalized = sum(1 for i in explore_indices if group[i].get("query_penalty", 0) > 0)
            print(
                f"[contrastive_choice] group={g} EXPLORE_BETTER "
                f"explore_avg={explore_avg:.2f} direct_avg={direct_avg:.2f} diff={diff:+.2f} "
                f"n_exp={len(explore_indices)} n_dir={len(direct_indices)}"
                f" query_penalized={n_query_penalized}",
                flush=True,
            )

        elif diff <= -threshold:
            for i in direct_indices:
                o = group[i]
                o["s_c"] = 1.0
                o["score"] = round(max(0.0, min(1.0, o["score"] + s_c_weight * 1.0)), 4)
                o["contrastive_label"] = "direct_rewarded"
                outputs[g * rollout_n + i] = o
            for i in explore_indices:
                o = group[i]
                o["s_c"] = 0.0
                o["contrastive_label"] = "explore_punished"
                outputs[g * rollout_n + i] = o
            print(
                f"[contrastive_choice] group={g} DIRECT_BETTER "
                f"explore_avg={explore_avg:.2f} direct_avg={direct_avg:.2f} diff={diff:+.2f} "
                f"n_exp={len(explore_indices)} n_dir={len(direct_indices)}",
                flush=True,
            )

        else:
            for i in range(len(group)):
                group[i]["contrastive_label"] = "neutral"
                group[i]["s_c"] = 0.0
                outputs[g * rollout_n + i] = group[i]
            print(
                f"[contrastive_choice] group={g} NEUTRAL "
                f"explore_avg={explore_avg:.2f} direct_avg={direct_avg:.2f} diff={diff:+.2f}",
                flush=True,
            )

    if reuse_eval_count:
        print(
            f"[contrastive_choice] skill_reuse_eval: n={reuse_eval_count} "
            f"total={reuse_eval_time:.1f}s avg={reuse_eval_time/reuse_eval_count:.1f}s",
            flush=True,
        )
    if query_penalty_eval_count:
        print(
            f"[contrastive_choice] query_penalty: n={query_penalty_eval_count} "
            f"total={query_penalty_eval_time:.1f}s avg={query_penalty_eval_time/query_penalty_eval_count:.1f}s",
            flush=True,
        )

    for o in outputs:
        o.pop("_generated_skills", None)
        o.pop("_similar_questions", None)
        o.pop("_page_visit_blocks", None)
        o.pop("_question", None)
        o.pop("_ground_truth", None)
        o.pop("_conversation_messages", None)

    return outputs


# ---------------------------------------------------------------------------
# Skill reuse evaluation (s_reuse)
# ---------------------------------------------------------------------------

_REUSE_INFRA: dict[str, Any] | None = None
_REUSE_INFRA_LOCK = __import__("threading").Lock()

_REUSE_EVAL_MAX_ROUNDS = int(os.environ.get("SKILL_REUSE_EVAL_MAX_ROUNDS", "6"))
_REUSE_EVAL_TEMPERATURE = float(os.environ.get("SKILL_REUSE_EVAL_TEMPERATURE", "0.7"))
_REUSE_EVAL_TIMEOUT = float(os.environ.get("SKILL_REUSE_EVAL_TIMEOUT", "120"))
_REUSE_MAX_TOOL_RESPONSE = 4096


def _get_reuse_eval_infra() -> dict[str, Any]:
    global _REUSE_INFRA
    with _REUSE_INFRA_LOCK:
        if _REUSE_INFRA is not None:
            return _REUSE_INFRA

    from evaluation.cascade_judge import load_config
    from tools.code_executor import CodeExecutor

    cfg = load_config()
    code_executor = CodeExecutor.from_config(cfg)
    profile = os.environ.get("SKILL_REUSE_LLM_PROFILE", "frozen_qwen3_8b_local")
    baseline_path = os.environ.get("SKILL_REUSE_BASELINE_PATH", "")
    baselines: dict[str, float] = {}
    if baseline_path:
        try:
            with open(baseline_path, "r") as f:
                baselines = {str(k): float(v) for k, v in json.load(f).items()}
            print(f"[skill_reuse_eval] loaded {len(baselines)} precomputed baselines from {baseline_path}", flush=True)
        except Exception as e:
            print(f"[skill_reuse_eval] failed to load baselines from {baseline_path}: {e}", flush=True)

    _smoke_timeout = float(os.environ.get("SKILL_REUSE_SMOKE_TIMEOUT", "30"))
    _smoke_prompt = [{"role": "user", "content": "Reply with exactly: PING_OK"}]
    print(f"[skill_reuse_eval] smoke-testing frozen model profile={profile} timeout={_smoke_timeout}s ...", flush=True)
    try:
        _smoke_resp = sync_chat_messages(profile, _smoke_prompt, temperature=0.0, timeout=_smoke_timeout)
    except Exception as _smoke_err:
        _smoke_resp = ""
        print(f"[skill_reuse_eval] smoke-test exception: {_smoke_err}", flush=True)
    if not _smoke_resp or not _smoke_resp.strip():
        raise RuntimeError(
            f"[skill_reuse_eval] FATAL: frozen model smoke-test failed "
            f"(profile={profile}, timeout={_smoke_timeout}s). "
            f"Check that the frozen model service is running and reachable. Aborting."
        )
    print(f"[skill_reuse_eval] smoke-test PASSED: {_smoke_resp.strip()[:80]}", flush=True)

    infra = {"code_executor": code_executor, "profile": profile, "baselines": baselines}
    with _REUSE_INFRA_LOCK:
        _REUSE_INFRA = infra
    return infra


def _build_skill_workflow_text(skills: list[dict[str, Any]]) -> str:
    blocks = []
    for s in skills:
        name = s.get("skill_name", "unnamed")
        wf = (skill_workflow(s) or "").strip()
        if wf:
            blocks.append(f"### Skill: {name}\nWorkflow:\n{wf}")
    return "\n\n".join(blocks)


def _build_skill_code_block(skills: list[dict[str, Any]]) -> str:
    parts = []
    for s in skills:
        c = (skill_code(s) or "").strip()
        if c:
            parts.append(c)
    return "\n\n".join(parts)


def _build_code_only_initial_prompt(
    question_text: str,
    injected_skill_workflow: str | None,
    injected_skill_code: str | None = None,
    reference_only: bool = False,
) -> str:
    if injected_skill_workflow and reference_only:
        code_block = (
            f"\nReference code (NOT loaded — for inspiration only):\n```python\n{injected_skill_code}\n```\n"
            if injected_skill_code else ""
        )
        skill_section = (
            f"\n## Reference Skill (guidance only — NOT activated):\n"
            f"{injected_skill_workflow}\n"
            f"{code_block}\n"
            "Treat the workflow and code above as REFERENCE ONLY. They may be too specific to the "
            "original problem and are NOT loaded into the execution environment. Adapt the ideas to "
            "this question and write your own code in <CODE>...</CODE>.\n"
        )
    elif injected_skill_workflow:
        skill_section = (
            f"\n## Available Skill (already loaded — workflow and code are active):\n"
            f"{injected_skill_workflow}\n\n"
            "The skill's code functions are pre-loaded; call them directly inside <CODE>...</CODE> "
            "without re-defining or importing.\n"
        )
    else:
        skill_section = ""
    return (
        "You are an intelligent assistant solving a problem. Think step-by-step in plain text, "
        "then output EXACTLY ONE action tag at the end. You will receive feedback after each action.\n\n"
        f"## Question:\n{question_text}\n"
        f"{skill_section}"
        "Available actions (choose exactly one per turn):\n"
        "1) Run Python code: <CODE>your code here</CODE>\n"
        "2) Final answer: <ANSWER>final answer (or option letter for MC)</ANSWER>\n"
    )


def _run_single_sub_rollout_sync(
    question_text: str,
    skill_workflow_text: str | None,
    skill_code_block: str | None,
    profile: str,
    code_executor: Any,
) -> dict[str, Any]:
    env_ref = os.environ.get("SKILL_REFERENCE_ONLY", "").strip().lower()
    reference_only = env_ref in ("1", "true", "yes") and bool(skill_workflow_text)
    initial_prompt = _build_code_only_initial_prompt(
        question_text=question_text,
        injected_skill_workflow=skill_workflow_text,
        injected_skill_code=skill_code_block if reference_only else None,
        reference_only=reference_only,
    )
    messages: list[dict[str, str]] = [{"role": "user", "content": initial_prompt}]

    code_context: list[str] = []
    if skill_code_block and not reference_only:
        try:
            compile(skill_code_block, "<skill>", "exec")
            code_context.append(skill_code_block)
        except SyntaxError:
            pass

    final_pred = ""
    for _ in range(_REUSE_EVAL_MAX_ROUNDS):
        text = sync_chat_messages(
            profile,
            messages,
            temperature=_REUSE_EVAL_TEMPERATURE,
            timeout=_REUSE_EVAL_TIMEOUT,
        )
        if not text:
            break
        messages.append({"role": "assistant", "content": text})
        step = parse_execution_step(text)

        if step["type"] == "answer":
            final_pred = step.get("answer", "").strip()
            break

        if step["type"] == "run_code":
            code = step.get("run_code", "").strip()
            if not code or code_executor is None:
                feedback = "Code execution unavailable. Use <ANSWER>...</ANSWER> to answer."
            else:
                full_code = "\n\n".join(code_context + [f"# --- User code ---\n{code}"]) if code_context else code
                result = code_executor.execute(full_code)
                if result["success"]:
                    out_text = result["stdout"].strip() or "(no output, remember to print() the result)"
                    feedback = (
                        f"Code feedback:\nOutput:\n{out_text}\n\n"
                        "Continue with <CODE>...</CODE> or <ANSWER>...</ANSWER>."
                    )
                else:
                    feedback = (
                        f"Code feedback:\nstdout: {result['stdout']}\nstderr: {result['stderr']}\n\n"
                        "Continue with <CODE>...</CODE> or <ANSWER>...</ANSWER>."
                    )
        else:
            feedback = "Format error. Use exactly one of <CODE>...</CODE> or <ANSWER>...</ANSWER>."

        if len(feedback) > _REUSE_MAX_TOOL_RESPONSE:
            feedback = feedback[:_REUSE_MAX_TOOL_RESPONSE] + "\n...[truncated]"
        messages.append({"role": "user", "content": feedback})

    if not final_pred:
        for m in reversed(messages):
            if m["role"] == "assistant":
                final_pred = extract_answer_from_tag(m["content"])
                if final_pred:
                    break

    return {"prediction": final_pred, "messages": messages}


def _evaluate_skill_reuse_sync(
    generated_skills: list[dict[str, Any]],
    similar_questions: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """On-demand skill reuse eval (sync). Runs sub-rollouts with frozen model.

    Returns (score [0,1], details).
    """
    import time as _time

    if not generated_skills:
        return 0.5, {"reason": "no_skill_generated"}
    similar_questions = list(similar_questions or [])[:3]
    if not similar_questions:
        return 0.5, {"reason": "no_similar_questions"}

    infra = _get_reuse_eval_infra()
    code_executor = infra["code_executor"]
    profile = infra["profile"]
    baselines = infra["baselines"]

    skill_workflow_text = _build_skill_workflow_text(generated_skills)
    skill_code_block = _build_skill_code_block(generated_skills)

    total_timeout = float(os.environ.get("SKILL_REUSE_EVAL_TOTAL_TIMEOUT", "1500"))
    deadline = _time.time() + total_timeout

    per_q: list[dict[str, Any]] = []
    n_timed_out = 0
    for sq in similar_questions:
        question_text = str(sq.get("question") or "").strip()
        q_id = str(sq.get("id", ""))
        gt = str(sq.get("answer") or "").strip()

        has_precomputed = q_id in baselines
        b_rate = float(baselines[q_id]) if has_precomputed else 0.0

        remaining = deadline - _time.time()
        if remaining <= 0:
            per_q.append({
                "id": q_id,
                "baseline_pass_rate": round(b_rate, 4),
                "baseline_precomputed": has_precomputed,
                "with_skill_pass_rate": round(b_rate, 4),
                "delta": 0.0,
                "timed_out": True,
            })
            n_timed_out += 1
            continue

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _run_single_sub_rollout_sync,
                    question_text=question_text,
                    skill_workflow_text=skill_workflow_text,
                    skill_code_block=skill_code_block,
                    profile=profile,
                    code_executor=code_executor,
                )
                result = future.result(timeout=remaining)
            prediction = result.get("prediction", "")
            if prediction and gt:
                score_val, _ = cascade_answer_judge(
                    question=question_text,
                    ground_truth=gt,
                    prediction=prediction,
                )
            else:
                score_val = 0.0
            w_rate = score_val
            timed_out = False
        except Exception as e:
            is_timeout = "TimeoutError" in type(e).__name__ or "timed out" in str(e).lower()
            if is_timeout:
                print(f"[skill_reuse_eval] sub-rollout timed out for q_id={q_id}", flush=True)
                n_timed_out += 1
            else:
                print(f"[skill_reuse_eval] sub-rollout failed for q_id={q_id}: {e}", flush=True)
            w_rate = b_rate
            timed_out = is_timeout

        per_q.append({
            "id": q_id,
            "baseline_pass_rate": round(b_rate, 4),
            "baseline_precomputed": has_precomputed,
            "with_skill_pass_rate": round(w_rate, 4),
            "delta": round(w_rate - b_rate, 4),
            "timed_out": timed_out,
        })

    valid_q = [q for q in per_q if not q.get("timed_out")]
    if not valid_q:
        return 0.5, {
            "reason": "all_timed_out",
            "k_evaluated": 0,
            "n_timed_out": n_timed_out,
            "k_total": len(per_q),
        }

    deltas = [q["delta"] for q in valid_q]
    k = len(deltas)
    raw = sum(deltas) / k if k else 0.0

    # Map δ̄ ∈ [-1, 1] to s_reuse ∈ [0, 1] via clip((δ̄ + 1) / 2)
    score = max(0.0, min(1.0, (raw + 1.0) / 2.0))
    return score, {
        "reason": "evaluated",
        "raw_mean_delta": round(raw, 4),
        "k_evaluated": k,
        "n_timed_out": n_timed_out,
        "k_total": len(per_q),
        "per_q_deltas": [round(d, 4) for d in deltas],
        "per_q_baseline": [q["baseline_pass_rate"] for q in valid_q],
        "per_q_with_skill": [q["with_skill_pass_rate"] for q in valid_q],
    }


# ---------------------------------------------------------------------------
# Query penalty (p_q): 2-dimension judge (Query Abstraction + Evidence Gain)
# ---------------------------------------------------------------------------

_QUERY_JUDGE_INFRA: dict[str, Any] | None = None
_QUERY_JUDGE_INFRA_LOCK = __import__("threading").Lock()
_QUERY_JUDGE_PROFILE_CURSOR = 0
_QUERY_JUDGE_TIMEOUT = float(os.environ.get("QUERY_JUDGE_TIMEOUT", "60"))
_QUERY_JUDGE_TOTAL_TIMEOUT = float(os.environ.get("QUERY_JUDGE_TOTAL_TIMEOUT", "300"))

_QUERY_JUDGE_PROMPT = """You are a query-quality judge for an AI agent that answers questions using web search. The agent has generated a set of Google search queries during exploration. Your job is to evaluate whether these queries retrieve useful evidence to solve the question from the agent's current uncertainty. Do NOT judge whether the final answer is correct; do NOT use the ground-truth answer.

Inputs.
- Question: {question}
- Candidate options (if any): {options}
- Agent goal: {exploration_goal}
- Agent reasoning before query: {agent_reasoning_before_query}
- Queries to judge: {queries_json}
- Search snippets returned: {search_snippets}

Scoring dimensions. Score the query evaluation on the following 2 dimensions. Each dimension must be an integer score: 0, 1, 2, or 3.

1. Query Abstraction. Whether the query searches for reusable principles, formulas, or workflows, rather than over-fitting to specific instance numbers or searching vague broad topics.
- 0 = Instance-Overfitted: query contains specific problem numbers or local variables (e.g. "calculate X when a=3, b=5"), directly fishing for the answer.
- 1 = Broad Topic: query is too vague, using only top-level domain keywords (e.g. "physics thermodynamics").
- 2 = Concrete Target: targets the right uncertainty, but still carries specific numbers or instance context from the problem.
- 3 = De-instanced Precision: strips away problem-specific numbers/entities and uses precise domain terminology to search for reusable formulas or standard methods.

2. Evidence Gain. Whether the returned search snippets eliminate confusion and provide verifiable, non-redundant evidence.
- 0 = Noisy/Irrelevant: snippets are empty, noisy, or completely useless.
- 1 = Redundant Background: snippets only give common background knowledge the agent already knew.
- 2 = Partial Support: snippets provide useful formulas or rules that narrow down the space.
- 3 = Crucial Gain: snippets provide direct conceptual proof or clear counter-evidence to resolve the core uncertainty.

Output (return ONLY valid JSON).
{{
  "dimension_scores": {{
    "query_abstraction": <integer 0-3>,
    "evidence_gain": <integer 0-3>
  }},
  "rationale": "1-3 sentences explaining the main strengths or weaknesses."
}}"""


def _get_query_judge_infra() -> dict[str, Any] | None:
    global _QUERY_JUDGE_INFRA
    with _QUERY_JUDGE_INFRA_LOCK:
        if _QUERY_JUDGE_INFRA is not None:
            return _QUERY_JUDGE_INFRA

    profiles_raw = os.environ.get("QUERY_JUDGE_LLM_PROFILE", "")
    if not profiles_raw:
        profiles_raw = os.environ.get("SKILL_QUALITY_PROFILES", "qwen37-max,qwen37-max2")
    profiles = [p.strip() for p in profiles_raw.split(",") if p.strip()]
    if not profiles:
        print("[query_penalty] No LLM profile configured, query penalty disabled", flush=True)
        return None

    _smoke_timeout = float(os.environ.get("QUERY_JUDGE_SMOKE_TIMEOUT", "30"))
    usable_profiles = []
    for profile in profiles:
        print(f"[query_penalty] smoke-testing judge profile={profile} timeout={_smoke_timeout}s ...", flush=True)
        try:
            _smoke_resp = sync_chat_messages(profile, [{"role": "user", "content": "Reply with exactly: PING_OK"}], temperature=0.0, timeout=_smoke_timeout)
        except Exception as _smoke_err:
            _smoke_resp = ""
            print(f"[query_penalty] smoke-test exception profile={profile}: {_smoke_err}", flush=True)
        if not _smoke_resp or not _smoke_resp.strip():
            print(f"[query_penalty] smoke-test FAILED profile={profile}, skipping.", flush=True)
            continue
        print(f"[query_penalty] smoke-test PASSED profile={profile}: {_smoke_resp.strip()[:80]}", flush=True)
        usable_profiles.append(profile)
    if not usable_profiles:
        print("[query_penalty] all judge profile smoke-tests failed, query penalty will be skipped.", flush=True)
        return None

    infra = {"profiles": usable_profiles}
    with _QUERY_JUDGE_INFRA_LOCK:
        _QUERY_JUDGE_INFRA = infra
    return infra


def _extract_query_events_from_conversation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    current_goal = ""
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content") or ""

        goal_patterns = [
            r"Current goal:\s*(.*?)(?:\n\s*\n|\nRules:|\Z)",
            r"Exploration goal:\s*(.*?)(?:\nSearch results|\n\s*\n|\Z)",
        ]
        for pat in goal_patterns:
            m = re.search(pat, content, flags=re.DOTALL)
            if m:
                current_goal = " ".join(m.group(1).strip().split())
                break

        if role != "assistant" or "<QUERY" not in content.upper():
            continue

        query_matches = list(re.finditer(r"<QUERY>\s*(.*?)\s*</QUERY>", content, flags=re.IGNORECASE | re.DOTALL))
        for qm in query_matches:
            raw = qm.group(1).strip()
            queries = _parse_query_content(raw)
            if not queries:
                continue

            reasoning = content[: qm.start()]
            reasoning = re.sub(r"</?think>", "", reasoning, flags=re.IGNORECASE).strip()
            reasoning = re.sub(r"<[^>]+>", "", reasoning, flags=re.DOTALL).strip()
            reasoning = " ".join(reasoning.split())
            max_reasoning_chars = int(os.environ.get("QUERY_JUDGE_MAX_REASONING_CHARS", "3000"))
            if len(reasoning) > max_reasoning_chars:
                reasoning = reasoning[:max_reasoning_chars]

            search_snippets = ""
            for nxt in messages[i + 1:]:
                nxt_content = nxt.get("content") or ""
                if nxt.get("role") == "assistant" and "<QUERY" in nxt_content.upper():
                    break
                marker = "Search results for queries:"
                idx = nxt_content.find(marker)
                if idx != -1:
                    search_snippets = nxt_content[idx:].strip()
                    break

            events.append({
                "event_idx": len(events),
                "queries": queries,
                "exploration_goal": current_goal,
                "agent_reasoning_before_query": reasoning,
                "search_snippets": search_snippets,
            })
    return events


def _extract_options_from_question(question: str) -> str:
    matches = list(re.finditer(r"(?ms)^([A-J])\.\s*(.*?)(?=\n[A-J]\.\s|\Z)", question or ""))
    if not matches:
        return "(No explicit candidates parsed.)"
    lines = []
    for m in matches:
        option_text = " ".join(m.group(2).strip().split())
        lines.append(f"{m.group(1)}. {option_text}")
    return "\n".join(lines)


def _parse_query_judge_response(text: str | None) -> dict[str, Any]:
    if not text:
        return {"dimension_scores": None, "parse_failed": True}
    cleaned = text.strip()
    if "<think>" in cleaned:
        idx = cleaned.rfind("</think>")
        if idx != -1:
            cleaned = cleaned[idx + len("</think>"):].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    try:
        obj = json.loads(cleaned[start:end])
    except Exception:
        return {"dimension_scores": None, "parse_failed": True}
    raw_scores = obj.get("dimension_scores") or {}
    scores = {}
    for dim in ("query_abstraction", "evidence_gain"):
        try:
            value = max(0, min(3, int(raw_scores.get(dim))))
        except Exception:
            value = None
        scores[dim] = value
    parse_failed = any(scores[d] is None for d in ("query_abstraction", "evidence_gain"))
    return {"dimension_scores": scores, "parse_failed": parse_failed}


def _compute_query_penalty_for_rollout(
    question: str,
    messages: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Judge query quality and return penalty p_q = ρ·(𝟙(c₁≤1) + 𝟙(c₂≤1)).

    Returns (negative_penalty, details).
    """
    infra = _get_query_judge_infra()
    if infra is None:
        return 0.0, {"reason": "no_infra"}

    events = _extract_query_events_from_conversation(messages)
    if not events:
        return 0.0, {"reason": "no_query_events", "n_events": 0}
    max_events = max(1, int(os.environ.get("QUERY_JUDGE_MAX_EVENTS_PER_ROLLOUT", "1")))
    events = events[:max_events]

    global _QUERY_JUDGE_PROFILE_CURSOR
    profiles = infra.get("profiles") or []
    if not profiles:
        return 0.0, {"reason": "no_profiles"}
    with _QUERY_JUDGE_INFRA_LOCK:
        profile = profiles[_QUERY_JUDGE_PROFILE_CURSOR % len(profiles)]
        _QUERY_JUDGE_PROFILE_CURSOR += 1
    options = _extract_options_from_question(question)
    max_q_chars = int(os.environ.get("QUERY_JUDGE_MAX_QUESTION_CHARS", "6000"))
    max_snippet_chars = int(os.environ.get("QUERY_JUDGE_MAX_SNIPPET_CHARS", "6000"))
    max_reasoning_chars = int(os.environ.get("QUERY_JUDGE_MAX_REASONING_CHARS", "3000"))

    rho = QUERY_PENALTY_RHO
    total_penalty = 0.0
    all_details = []

    for event in events:
        q_text = question[:max_q_chars] if len(question) > max_q_chars else question
        prompt = _QUERY_JUDGE_PROMPT.format(
            question=q_text,
            options=options[:3000] if len(options) > 3000 else options,
            exploration_goal=(event.get("exploration_goal") or "(not found)")[:1200],
            agent_reasoning_before_query=(event.get("agent_reasoning_before_query") or "(not found)")[:max_reasoning_chars],
            queries_json=json.dumps(event.get("queries", []), ensure_ascii=False, indent=2),
            search_snippets=(event.get("search_snippets") or "(not available)")[:max_snippet_chars],
        )

        try:
            resp = sync_chat_messages(
                profile,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=_QUERY_JUDGE_TIMEOUT,
            )
            judged = _parse_query_judge_response(resp)
        except Exception as e:
            print(f"[query_penalty] judge call failed: {e}", flush=True)
            judged = {"dimension_scores": None, "parse_failed": True}

        scores = judged.get("dimension_scores") or {}
        c1 = scores.get("query_abstraction")
        c2 = scores.get("evidence_gain")

        # p_q = ρ · (𝟙(c₁ ≤ 1) + 𝟙(c₂ ≤ 1))
        event_penalty = 0.0
        if c1 is not None and c1 <= 1:
            event_penalty += rho
        if c2 is not None and c2 <= 1:
            event_penalty += rho
        total_penalty += event_penalty

        all_details.append({
            "event_idx": event["event_idx"],
            "queries": event.get("queries", []),
            "query_abstraction_score": c1,
            "evidence_gain_score": c2,
            "event_penalty": round(event_penalty, 4),
            "parse_failed": judged.get("parse_failed", False),
        })

    total_penalty = min(total_penalty, 0.3)

    return -total_penalty, {
        "reason": "judged",
        "n_events": len(events),
        "query_abstraction_score": all_details[0].get("query_abstraction_score") if all_details else None,
        "evidence_gain_score": all_details[0].get("evidence_gain_score") if all_details else None,
        "total_penalty": round(total_penalty, 4),
        "event_details": all_details,
    }


# ---------------------------------------------------------------------------
# Skill quality score (s_ground, from agent_loop's skill quality judge)
# ---------------------------------------------------------------------------

def _compute_skill_quality_score(
    extra_info: dict[str, Any],
    version_cfg: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Read agent_loop's skill_quality_eval_results.

    Extracts grounding score for s_ground computation.
    Returns the raw grounding score (1-10) and details.
    """
    if not version_cfg.get("use_skill_quality", False):
        return 5.0, {"reason": "disabled"}

    default_score = 5.0
    eval_results = extra_info.get("skill_quality_eval_results")
    if not isinstance(eval_results, dict):
        return default_score, {"reason": "no_skill_or_skipped", "n_skills": 0}
    if eval_results.get("skipped"):
        return default_score, {
            "reason": str(eval_results.get("reason") or "skipped"),
            "n_skills": 0,
        }

    per_skill = eval_results.get("per_skill") or []
    grounding_vals = [int(s.get("grounding")) for s in per_skill if isinstance(s.get("grounding"), (int, float))]
    grounding_mean = (sum(grounding_vals) / len(grounding_vals)) if grounding_vals else default_score

    any_cs_error = False

    return grounding_mean, {
        "reason": str(eval_results.get("reason") or "judged"),
        "n_skills": len(per_skill),
        "grounding_mean": round(grounding_mean, 4),
    }


# ---------------------------------------------------------------------------
# Per-sample reward computation
# ---------------------------------------------------------------------------

def _compute_single(data_source: Any, solution_str: str, ground_truth: Any, extra_info: dict[str, Any] | None,
                    _sample_timings: list | None = None) -> dict[str, Any]:
    import time as _time
    _t_single_start = _time.time()
    extra_info = dict(extra_info or {})
    skillset_meta = dict(extra_info.get("skillset_meta") or {})

    stage = int(extra_info.get("stage", skillset_meta.get("stage", 0)) or 0)
    version_cfg = REWARD_CONFIG

    assistant_text = _extract_assistant_text(solution_str, extra_info)
    actions = _get_actions(extra_info, assistant_text)
    flow_analysis = _analyze_explore_flow(actions)

    api_stats: list[dict[str, Any]] = []

    _t_ans = _time.time()
    answer_score, answer_judge_raw = _answer_score(
        assistant_text,
        ground_truth,
        extra_info,
        data_source=data_source,
        api_stats=api_stats,
    )
    _answer_time = _time.time() - _t_ans
    format_score = _format_score(assistant_text, extra_info=extra_info)
    generated_skills = _get_generated_skills(extra_info, assistant_text)
    flow_penalty = flow_analysis["flow_penalty"] if version_cfg["use_flow_penalty"] else 0.0
    explore_score = max(0.0, 1.0 - flow_penalty)

    # Grounding score (raw, 1-10) from teacher LLM judge in agent_loop
    grounding_raw, skill_quality_details = _compute_skill_quality_score(
        extra_info=extra_info,
        version_cfg=version_cfg,
    )

    total = answer_score * float(version_cfg.get("s_a", 0.7)) - flow_penalty
    total = max(0.0, min(1.0, total))

    print(
        "[search2skill_reward] "
        f"stage={stage} "
        f"actions={actions} "
        f"flow_penalty={flow_penalty:.3f} "
        f"answer={answer_score:.3f} "
        f"format={format_score:.3f} "
        f"grounding_raw={grounding_raw:.1f} "
        f"score={total:.3f}",
        flush=True,
    )

    api_by_profile: dict[str, dict[str, int]] = {}
    for stat in api_stats:
        key = stat["profile"]
        if key not in api_by_profile:
            api_by_profile[key] = {"total": 0, "success": 0}
        api_by_profile[key]["total"] += 1
        if stat["success"]:
            api_by_profile[key]["success"] += 1

    if _sample_timings is not None:
        _sample_timings.append({
            "answer_judge": _answer_time,
            "other": max(0.0, _time.time() - _t_single_start - _answer_time),
        })

    similar_questions = list(extra_info.get("similar_questions") or [])
    page_visit_blocks = list(extra_info.get("page_visit_blocks") or [])
    question = str(extra_info.get("question") or "").strip()
    ground_truth_str = str(ground_truth or "").strip()

    return {
        "score": round(total, 4),
        "answer_score": round(answer_score, 4),
        "format_score": round(format_score, 4),
        "explore_score": round(explore_score, 4),
        "s_c": 0.0,
        "s_ground": 0.0,
        "contrastive_label": "",
        "query_penalty": 0.0,
        "query_penalty_dim_scores": {},
        "flow_penalty": round(flow_penalty, 4),
        "flow_violation_count": len(flow_analysis["violations"]),
        "skill_reuse_score": 0.0,
        "skill_reuse_raw_delta": 0.0,
        "k_similar_evaluated": 0,
        "skill_reuse_reason": "deferred",
        "skill_quality_grounding_raw": round(grounding_raw, 4),
        "skill_quality_n_skills": int(skill_quality_details.get("n_skills", 0)) or len(generated_skills),
        "skill_quality_reason": str(skill_quality_details.get("reason", "")),
        "skill_generated": int(len(generated_skills) > 0),
        "stage": stage,
        "used_explore": flow_analysis["has_explore"],
        "compliant_explore": flow_analysis["has_valid_explore_path"],
        "explore_required": bool(skillset_meta.get("explore_required", extra_info.get("explore_required", False))),
        "answer_judge_raw": answer_judge_raw,
        "flow_violations": ",".join(flow_analysis["violations"]),
        "api_calls_total": len(api_stats),
        "api_calls_success": sum(1 for s in api_stats if s["success"]),
        "api_calls_failed": len(api_stats) - sum(1 for s in api_stats if s["success"]),
        "api_stats_by_profile": api_by_profile,
        "_generated_skills": generated_skills,
        "_similar_questions": similar_questions,
        "_page_visit_blocks": page_visit_blocks,
        "_question": question,
        "_ground_truth": ground_truth_str,
        "_conversation_messages": extra_info.get("conversation_messages") or [],
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _is_single_letter(text: str, allowed: str = "ABCDEFGHIJ") -> str | None:
    if not text:
        return None
    stripped = str(text).strip()
    if len(stripped) <= 2 and stripped[0].upper() in allowed:
        return stripped[0].upper()
    return None


def _answer_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    data_source: Any = None,
    api_stats: list | None = None,
) -> tuple[float, str]:
    prediction = extract_final_answer(solution_str)
    if not prediction:
        return 0.0, "missing_answer_tag"

    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target") or ground_truth.get("answer") or ground_truth.get("ground_truth")
    question = str(extra_info.get("question") or "").strip()
    gt = str(ground_truth or "").strip().upper()

    if len(gt) == 1 and gt in "ABCDEFGHIJ":
        pred_letter = _is_single_letter(prediction)
        if pred_letter:
            correct = pred_letter == gt
            score = 1.0 if correct else 0.0
            reason = f"letter_rule:{'correct' if correct else 'incorrect'}|pred={pred_letter}|gt={gt}"
            return score, reason

    return cascade_answer_judge(
        question=question,
        ground_truth=str(ground_truth or "").strip(),
        prediction=prediction,
        api_stats=api_stats,
    )


def _paired_tag_count(solution_str: str, tag_name: str) -> int:
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    return len(re.findall(pattern, solution_str or "", re.DOTALL | re.IGNORECASE))


def _extract_think_blocks(text: str) -> list[str]:
    return re.findall(r"<think>(.*?)</think>", text or "", re.DOTALL | re.IGNORECASE)


def _strip_think_blocks(text: str) -> str:
    if not text or not re.search(r"<think>", text, re.IGNORECASE):
        return text or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _normalize_action_name(action: str) -> str:
    normalized = str(action or "").strip().lower()
    if normalized == "skill_call":
        return "activate_skill"
    return normalized


def _extract_actions_from_tags(assistant_text: str) -> list[str]:
    matches = re.findall(
        r"<(ANSWER|ACTIVATE_SKILL|CODE|EXPLORE|QUERY|PAGE_VISIT|GENERATE_SKILL)>",
        assistant_text or "",
        re.IGNORECASE,
    )
    return [_normalize_action_name(match) for match in matches]


def _get_actions(extra_info: dict[str, Any], assistant_text: str) -> list[str]:
    raw_actions = extra_info.get("actions")
    if raw_actions:
        return [_normalize_action_name(action) for action in raw_actions if action]
    return _extract_actions_from_tags(assistant_text)


def _get_generated_skills(extra_info: dict[str, Any], assistant_text: str) -> list[dict[str, Any]]:
    generated = extra_info.get("generated_skills")
    if isinstance(generated, list) and generated:
        return [skill for skill in generated if isinstance(skill, dict)]
    return _extract_generated_skills(assistant_text)


def _analyze_explore_flow(actions: list[str]) -> dict[str, Any]:
    normalized_actions = [_normalize_action_name(action) for action in (actions or []) if action]
    violations: list[str] = []

    has_explore = False
    in_explore = False
    seen_query_in_session = False
    seen_page_visit_in_session = False
    valid_explore_sessions = 0
    illegal_exit_attempts = 0
    has_generated_skill = False

    flow_penalty = 0.0

    for action in normalized_actions:
        if action == "explore":
            has_explore = True
            if in_explore:
                violations.append("duplicate_explore")
                flow_penalty += 0.10
            in_explore = True
            seen_query_in_session = False
            seen_page_visit_in_session = False
            continue

        if action == "generate_skill":
            if not in_explore and (not has_generated_skill or valid_explore_sessions == 0):
                violations.append("generate_skill_without_explore")
                flow_penalty += 0.25
                has_generated_skill = True
                continue
            if in_explore:
                if not seen_query_in_session:
                    violations.append("generate_skill_without_query")
                    flow_penalty += 0.25
                if not seen_page_visit_in_session:
                    violations.append("generate_skill_without_page_visit")
                    flow_penalty += 0.20
                in_explore = False
                if seen_query_in_session and seen_page_visit_in_session:
                    valid_explore_sessions += 1
            has_generated_skill = True
            continue

        if action == "query":
            if in_explore:
                seen_query_in_session = True
            elif has_generated_skill:
                violations.append("post_generate_skill_query")
                flow_penalty += 0.10
            continue

        if action == "page_visit":
            if in_explore:
                seen_page_visit_in_session = True
            elif has_generated_skill:
                violations.append("post_generate_skill_page_visit")
                flow_penalty += 0.10
            continue

        if action in {"answer", "run_code", "activate_skill"} and in_explore:
            illegal_exit_attempts += 1
            violations.append(f"illegal_exit_attempt:{action}")
            flow_penalty += 0.10
            continue

    if in_explore:
        violations.append("unfinished_explore")
        flow_penalty += 0.20

    if illegal_exit_attempts >= 2:
        violations.append("repeated_illegal_exit_attempts")
        flow_penalty += min(0.05 * (illegal_exit_attempts - 1), 0.15)

    flow_penalty = max(0.0, min(0.60, flow_penalty))

    return {
        "actions": normalized_actions,
        "violations": violations,
        "flow_penalty": flow_penalty,
        "has_explore": has_explore,
        "has_valid_explore_path": valid_explore_sessions > 0,
    }


def _format_score(solution_str: str, extra_info: dict[str, Any] | None = None) -> float:
    messages = (extra_info or {}).get("conversation_messages")
    if messages:
        turns = [msg["content"] for msg in messages if msg.get("role") == "assistant" and msg.get("content")]
    else:
        turns = [solution_str] if solution_str else []

    if not turns:
        return 0.0

    n_well_formed = 0
    final_turn_has_answer = False

    for idx, turn_text in enumerate(turns):
        action_text = _strip_think_blocks(turn_text)

        paired_in_turn = 0
        all_balanced = True
        for tag_name in ACTION_TAGS:
            opens = len(re.findall(rf"<{tag_name}>", action_text, re.IGNORECASE))
            closes = len(re.findall(rf"</{tag_name}>", action_text, re.IGNORECASE))
            paired_in_turn += min(opens, closes)
            all_balanced = all_balanced and opens == closes

        if paired_in_turn == 1 and all_balanced:
            n_well_formed += 1

        if idx == len(turns) - 1 and _paired_tag_count(action_text, "ANSWER") >= 1:
            final_turn_has_answer = True

    score = n_well_formed / len(turns) * 0.60

    if final_turn_has_answer:
        score += 0.40

    return min(1.0, score)


def _extract_generated_skills(solution_str: str) -> list[dict[str, Any]]:
    matches = re.findall(r"<GENERATE_SKILL>(.*?)</GENERATE_SKILL>", solution_str or "", re.DOTALL | re.IGNORECASE)
    generated: list[dict[str, Any]] = []
    for raw in matches:
        parsed = parse_generate_skill(raw.strip())
        if parsed:
            generated.append(parsed)
    return generated
