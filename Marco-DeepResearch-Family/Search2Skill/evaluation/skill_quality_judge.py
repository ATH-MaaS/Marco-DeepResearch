"""Skill quality LLM-judge that scores a generated skill on grounding
and common-sense correctness using LLMClient.from_config.

Public entry point:
    await evaluate_skill_quality(
        question, ground_truth, student_knowledge,
        page_visits, explore_goals, generated_skills,
    ) -> {"score": float[0,1], "skipped": bool, "reason": str, "per_skill": [...]}
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from typing import Any, Optional

from llm.client import LLMClient

# cascade_judge already provides config + bg-loop helpers; reuse them.
from evaluation.cascade_judge import load_config


# ---------------------------- prompts ----------------------------

GROUNDING_PROMPT = """You are a strict skill-quality reviewer for a research project that uses LLM agents to "explore" the web and then distill a reusable skill (workflow + python helper functions).

You will rate ONE skill on a SINGLE dimension - Grounding - on an integer scale 1-10.

DIMENSION - Grounding
How much of the skill (workflow + code) is genuinely supported by the evidence the rollout retrieved during exploration?
  9-10 = Every important claim (formula, fact, constant, procedure, edge case) is supported by the evidence; no fabrication.
  7-8  = Most important claims are supported; a few minor items rely on prior knowledge.
  5-6  = Roughly half the skill is supported by evidence; the rest is prior knowledge or unverifiable.
  3-4  = Mostly prior knowledge dressed up as if evidence-derived; the evidence is barely used.
  1-2  = Skill fabricates content, misreads the evidence, or copies wrong numbers/units from the evidence.

Audit instructions:
- Pick the most important "core claims" the skill makes (formulas, numeric constants/conversion factors, named procedures, edge-case rules). Aim for 4-8 claims; skip pure boilerplate.
- For each, decide:
    "supported"  - clearly supported somewhere in the retrieved evidence (quote a SHORT snippet <= 25 words).
    "prior"      - not in the evidence at all; the model is using its own prior knowledge.
    "fabricated" - contradicts the evidence, misreads it, or copies a value with the wrong unit/dimension (briefly say how).
- Be specific about numbers/units.

Output STRICT JSON, NO prose outside:
{
  "score": <1-10>,
  "reason": "<= 120 words; explain the score using the audit findings>",
  "claims": [
    {"claim": "<paraphrased <= 25 words>",
     "verdict": "supported" | "prior" | "fabricated",
     "evidence_snippet": "<short quote <= 25 words; empty if not supported>",
     "note": "<= 25 words; optional clarification>"
    }
  ]
}
"""

# ---------------------------- LLMClient pool ----------------------------

_POOL_LOCK = threading.Lock()
_POOL: list[dict[str, Any]] | None = None
# Per-event-loop semaphores (asyncio.Semaphore must be created on the running loop).
_LOOP_SEMS: dict[int, list[asyncio.Semaphore]] = {}
_RR_INDEX = 0


def _profile_names() -> list[str]:
    raw = os.getenv("SKILL_QUALITY_PROFILES", "qwen37-max,qwen37-max2")
    return [p.strip() for p in raw.split(",") if p.strip()]


def _get_pool() -> list[dict[str, Any]]:
    """Lazy-load LLMClients (one per profile) from the shared exskill yaml."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            return _POOL
        cfg = load_config()
        pool: list[dict[str, Any]] = []
        for name in _profile_names():
            try:
                client = LLMClient.from_config(cfg, profile_name=name)
            except Exception as exc:
                print(f"[skill_quality_judge] WARN profile {name!r} unavailable: {exc}", flush=True)
                continue
            pool.append({"name": name, "client": client})
        if not pool:
            raise RuntimeError("[skill_quality_judge] no usable teacher profiles configured")
        _POOL = pool
        print(
            f"[skill_quality_judge] loaded {len(pool)} teacher profile(s): "
            f"{[p['name'] for p in pool]}",
            flush=True,
        )
        return pool


def _per_loop_sems() -> list[asyncio.Semaphore]:
    pool = _get_pool()
    loop = asyncio.get_event_loop()
    key = id(loop)
    sems = _LOOP_SEMS.get(key)
    if sems is None or len(sems) != len(pool):
        cap = int(os.getenv("SKILL_QUALITY_MAX_CONCURRENCY_PER_KEY", "10"))
        sems = [asyncio.Semaphore(cap) for _ in pool]
        _LOOP_SEMS[key] = sems
    return sems


async def _teacher_chat(messages: list[dict[str, Any]],
                        *, temperature: float = 0.2,
                        max_tokens: int = 8192,
                        max_retries: int = 3) -> str:
    """Round-robin between profiles, bounded by per-key Semaphore. Uses LLMClient.chat()."""
    global _RR_INDEX
    pool = _get_pool()
    sems = _per_loop_sems()
    n = len(pool)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        idx = _RR_INDEX % n
        _RR_INDEX = (_RR_INDEX + 1) % (n * 1024)
        prof = pool[idx]
        sem = sems[idx]
        try:
            async with sem:
                content = await prof["client"].chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            if content and content.strip():
                return content
            last_err = RuntimeError(f"empty response from {prof['name']}")
        except Exception as exc:
            last_err = exc
        await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"[skill_quality_judge] teacher chat failed: {last_err}")


# ---------------------------- JSON ----------------------------

def _extract_json(text: str) -> Any:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except Exception:
            pass
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
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
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"could not parse JSON from teacher: {text[:300]}")


# ---------------------------- per-skill judges ----------------------------

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _format_skill_payload(skill: dict[str, Any], max_chars: int) -> str:
    workflow = str(skill.get("workflow") or "")
    code = str(skill.get("code") or "")
    return _truncate(
        f"skill_name: {skill.get('skill_name','')}\n"
        f"use_when: {skill.get('use_when','')}\n"
        f"workflow:\n{workflow}\n\n"
        f"code:\n{code}\n",
        max_chars,
    )


async def _judge_grounding(
    question: str,
    ground_truth: str,
    explore_goals: list[str],
    page_visits: list[str],
    skill: dict[str, Any],
    max_evidence_chars: int,
    max_skill_chars: int,
) -> dict[str, Any]:
    evidence_blob = _truncate("\n\n=====\n\n".join(page_visits), max_evidence_chars)
    goals = "\n".join(f"- {g}" for g in explore_goals if str(g).strip()) or "(none)"
    user = (
        f"## Question\n{question}\n\n"
        f"## Ground truth\n{ground_truth}\n\n"
        f"## Explore goals (the rollout's own <EXPLORE>...</EXPLORE> queries)\n{goals}\n\n"
        f"## Evidence the rollout retrieved (raw page_visit blocks, joined)\n{evidence_blob}\n\n"
        f"## The skill the rollout produced\n{_format_skill_payload(skill, max_skill_chars)}"
    )
    raw = await _teacher_chat(
        [{"role": "system", "content": GROUNDING_PROMPT},
         {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=8192,
    )
    return _extract_json(raw)



def _coerce_score(v: Any, default: int = 5) -> int:
    try:
        x = int(round(float(v)))
    except Exception:
        return default
    return max(1, min(10, x))


async def _score_one_skill(
    question: str,
    ground_truth: str,
    student_knowledge: str,
    explore_goals: list[str],
    page_visits: list[str],
    skill: dict[str, Any],
    max_evidence_chars: int,
    max_skill_chars: int,
) -> dict[str, Any]:
    skip_grounding = os.getenv("SKILL_QUALITY_SKIP_GROUNDING", "").lower() in ("true", "1", "yes")

    if skip_grounding:
        d_ground = 5
        ground_reason = "skipped"
        audit: list[Any] = []
        errors: list[str] = []
    else:
        try:
            ground_res = await _judge_grounding(
                question, ground_truth, explore_goals, page_visits,
                skill, max_evidence_chars, max_skill_chars,
            )
        except Exception as exc:
            ground_res = exc

        d_ground = 5
        ground_reason = ""
        audit = []
        errors = []
        if isinstance(ground_res, Exception):
            errors.append(f"grounding:{type(ground_res).__name__}:{ground_res}")
        elif isinstance(ground_res, dict):
            d_ground = _coerce_score(ground_res.get("score"), 5)
            ground_reason = str(ground_res.get("reason") or "")[:600]
            audit = ground_res.get("claims") or []

    norm = max(0.0, min(1.0, float(d_ground) / 10.0))
    return {
        "skill_name": skill.get("skill_name") or "(unnamed)",
        "grounding": d_ground,
        "mean_raw": round(float(d_ground), 4),
        "norm": round(norm, 4),
        "grounding_reason": ground_reason,
        "grounding_audit": audit,
        "errors": errors,
    }


# ---------------------------- public entry ----------------------------

async def evaluate_skill_quality(
    question: str,
    ground_truth: str,
    student_knowledge: str,
    page_visits: list[str],
    explore_goals: list[str],
    generated_skills: list[dict[str, Any]],
    *,
    max_evidence_chars: int | None = None,
    max_skill_chars: int | None = None,
    no_skill_default: float | None = None,
) -> dict[str, Any]:
    """Score one rollout's generated skill(s).

    Returns:
      {
        "score": float in [0,1] (mean over per-skill norms),
        "skipped": bool,
        "reason": str,
        "elapsed_sec": float,
        "per_skill": [{...}]
      }
    """
    if max_evidence_chars is None:
        max_evidence_chars = int(os.getenv("SKILL_QUALITY_MAX_EVIDENCE_CHARS", "16000"))
    if max_skill_chars is None:
        max_skill_chars = int(os.getenv("SKILL_QUALITY_MAX_SKILL_CHARS", "6000"))
    if no_skill_default is None:
        no_skill_default = float(os.getenv("SKILL_QUALITY_NO_SKILL_DEFAULT", "0.5"))

    if not generated_skills:
        return {"score": no_skill_default, "skipped": True,
                "reason": "no_skill_generated", "per_skill": [], "elapsed_sec": 0.0}
    if not page_visits:
        return {"score": no_skill_default, "skipped": True,
                "reason": "no_page_visits", "per_skill": [], "elapsed_sec": 0.0}

    t0 = time.time()
    tasks = [
        _score_one_skill(
            question=question,
            ground_truth=ground_truth,
            student_knowledge=student_knowledge,
            explore_goals=explore_goals,
            page_visits=page_visits,
            skill=skill,
            max_evidence_chars=max_evidence_chars,
            max_skill_chars=max_skill_chars,
        )
        for skill in generated_skills
    ]
    per_skill_results = await asyncio.gather(*tasks, return_exceptions=True)

    final_per_skill: list[dict[str, Any]] = []
    norms: list[float] = []
    for skill, res in zip(generated_skills, per_skill_results):
        if isinstance(res, Exception):
            final_per_skill.append({
                "skill_name": skill.get("skill_name") or "(unnamed)",
                "errors": [f"{type(res).__name__}:{res}"],
                "grounding": 5,
                "mean_raw": 5.0, "norm": 0.5,
            })
            norms.append(0.5)
            continue
        final_per_skill.append(res)
        norms.append(float(res.get("norm", 0.5)))

    score = sum(norms) / max(len(norms), 1)
    return {
        "score": round(score, 4),
        "skipped": False,
        "reason": f"judged_{len(norms)}_skills",
        "elapsed_sec": round(time.time() - t0, 2),
        "per_skill": final_per_skill,
    }
