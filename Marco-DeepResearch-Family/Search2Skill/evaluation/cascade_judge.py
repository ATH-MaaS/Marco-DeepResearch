"""Cascade answer judge using LLMClient from llm.client.

Cascade flow:
  1. Exact match pre-check -> skip all model calls
  2. Primary model (config: rl.reward_judge_primary_profile) -> high confidence -> done
  3. Secondary model (config: rl.reward_judge_secondary_profile) -> final decision
  4. All tiers failed -> normalized text match fallback

Output is strictly binary: 0.0 (incorrect) or 1.0 (correct).
partially_correct / uncertain are treated as incorrect (0.0).

Configuration:
  All LLM profiles are read from config.yaml via LLMClient.from_config().
  Cascade profile names come from the ``rl`` section:
    - reward_judge_primary_profile   (default: "qwen3-30b-a3b")
    - reward_judge_secondary_profile (default: "qwen36-plus")

  Env var overrides:
    EXSKILL_JUDGE_CONFIDENCE_THRESHOLD  (default: 0.9)
"""

import asyncio
import json
import os
import re
import threading
from typing import Any, Optional

import yaml
from llm.client import LLMClient

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an expert evaluator tasked with determining whether a predicted answer matches the expected ground truth answer for a given question.

## Question
{question}

## Ground Truth Answer
{answer}

## Predicted Answer
{prediction}

## Evaluation Guidelines
1. **Semantic Equivalence**: Determine if the prediction conveys the same meaning as the ground truth answer, even if worded differently.
2. **Key Information**: Check if all critical facts, numbers, dates, names, or concepts from the ground truth are present in the prediction.
3. **Correctness**: The prediction should not contain factually incorrect information that contradicts the ground truth.
4. **Completeness**: The prediction should cover the essential aspects of the ground truth answer.

## Scoring Criteria
- **correct**: The prediction is semantically equivalent to the ground truth answer. Minor differences in wording, formatting, or additional non-contradictory details are acceptable.
- **incorrect**: The prediction is factually wrong, missing critical information, or contradicts the ground truth answer.
- **partially_correct**: The prediction contains some correct information but is incomplete or has minor errors.
- **uncertain**: You cannot confidently determine correctness due to ambiguity, lack of context, or the complexity exceeds your judgment capability. Use this sparingly.

## Output Format
Provide your evaluation in JSON format with the following fields:
- "judgment": One of "correct", "incorrect", "partially_correct", or "uncertain"
- "confidence": A float between 0 and 1 indicating your confidence in the judgment
- "reasoning": A brief explanation of your evaluation

**Return ONLY the JSON object, no additional text.**"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
)

CONFIDENCE_THRESHOLD = float(
    os.getenv("EXSKILL_JUDGE_CONFIDENCE_THRESHOLD", "0.95")
)

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

_CFG_CACHE: dict[str, dict[str, Any]] = {}
_CLIENT_CACHE: dict[str, LLMClient] = {}
_CACHE_LOCK = threading.Lock()

# Background event loop for sync -> async bridge
_BG_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BG_THREAD: Optional[threading.Thread] = None

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ``${ENV_VAR}`` references in a config dict/list."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                obj[key] = os.environ.get(value[2:-1], value)
            else:
                _resolve_env_vars(value)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                obj[idx] = os.environ.get(value[2:-1], value)
            else:
                _resolve_env_vars(value)
    return obj


def load_config() -> dict[str, Any]:
    """Load and cache the ExSkill YAML config (with env-var resolution)."""
    config_path = os.environ.get("EXSKILL_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    with _CACHE_LOCK:
        cached = _CFG_CACHE.get(config_path)
        if cached is not None:
            return cached

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg = _resolve_env_vars(cfg)

    with _CACHE_LOCK:
        _CFG_CACHE[config_path] = cfg
    return cfg


# ---------------------------------------------------------------------------
# LLMClient helpers
# ---------------------------------------------------------------------------


def _get_client(profile_name: str) -> LLMClient:
    """Get or create a cached LLMClient for the given config profile."""
    with _CACHE_LOCK:
        cached = _CLIENT_CACHE.get(profile_name)
        if cached is not None:
            return cached

    cfg = load_config()
    client = LLMClient.from_config(cfg, profile_name)

    with _CACHE_LOCK:
        _CLIENT_CACHE[profile_name] = client
    return client


def _ensure_bg_loop() -> asyncio.AbstractEventLoop:
    """Get (or create) a persistent background event loop for sync->async calls."""
    global _BG_LOOP, _BG_THREAD
    with _CACHE_LOCK:
        if _BG_LOOP is not None and _BG_LOOP.is_running():
            return _BG_LOOP
        _BG_LOOP = asyncio.new_event_loop()
        _BG_THREAD = threading.Thread(target=_BG_LOOP.run_forever, daemon=True)
        _BG_THREAD.start()
        return _BG_LOOP


def _llm_chat(client: LLMClient, prompt: str, temperature: float = 0.0) -> Optional[str]:
    """Synchronous wrapper around async ``LLMClient.chat()``.

    Returns the response string, or ``None`` if all retries failed.
    """
    loop = _ensure_bg_loop()
    coro = client.chat(prompt, temperature=temperature)
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=180)
    except Exception:
        return None


def sync_chat_messages(
    profile_name: str,
    messages: list,
    temperature: float = 0.7,
    timeout: float = 300,
) -> str:
    """Synchronous multi-turn chat via a named LLM profile.

    Like ``sync_chat`` but accepts a ``messages`` list instead of a single
    prompt string.  Used by reward-phase skill-reuse evaluation.
    """
    client = _get_client(profile_name)
    loop = _ensure_bg_loop()
    coro = client.chat(messages, temperature=temperature)
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout) or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def normalize_text(text: Any) -> str:
    """Lowercase, strip tags/whitespace, remove trailing punctuation."""
    value = str(text or "").strip().lower()
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,:;!?\n\t")


def extract_final_answer(solution_str: str) -> str:
    """Return content of the last ``<ANSWER>...</ANSWER>`` tag, or empty string."""
    matches = re.findall(
        r"<ANSWER>(.*?)</ANSWER>", solution_str or "", re.DOTALL | re.IGNORECASE
    )
    return matches[-1].strip() if matches else ""


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------


def judge_profiles() -> tuple[str, str]:
    """Return ``(primary_profile, secondary_profile)`` for cascade judge."""
    cfg = load_config()
    rl_cfg = cfg.get("rl", {})
    primary = os.environ.get(
        "EXSKILL_REWARD_JUDGE_PRIMARY_PROFILE",
        rl_cfg.get(
            "reward_judge_primary_profile",
            rl_cfg.get("reward_judge_profile", "qwen36-plus"),
        ),
    )
    secondary = os.environ.get(
        "EXSKILL_REWARD_JUDGE_SECONDARY_PROFILE",
        rl_cfg.get("reward_judge_secondary_profile", ""),
    )
    return primary, secondary


def reward_profiles() -> tuple[str, str]:
    """Return ``(judge_profile, similarity_profile)``."""
    cfg = load_config()
    rl_cfg = cfg.get("rl", {})
    judge_profile = os.environ.get(
        "EXSKILL_REWARD_JUDGE_PROFILE",
        rl_cfg.get("reward_judge_profile", "qwen36-plus"),
    )
    similarity_profile = os.environ.get(
        "EXSKILL_SIMILARITY_JUDGE_PROFILE",
        rl_cfg.get("similarity_judge_profile", "qwen36-plus"),
    )
    return judge_profile, similarity_profile


# ---------------------------------------------------------------------------
# Public sync_chat (used by reward for similarity scoring)
# ---------------------------------------------------------------------------


def sync_chat(
    profile_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_retries: int = 3,
    api_stats: list | None = None,
) -> str:
    """Synchronous single-turn chat using LLMClient.

    ``max_retries`` is accepted for backward compatibility but retries are
    handled internally by ``LLMClient``.
    """
    client = _get_client(profile_name)
    result = _llm_chat(client, prompt, temperature=temperature)
    if api_stats is not None:
        api_stats.append({
            "profile": profile_name,
            "model": client.default_model,
            "success": result is not None,
        })
    return result or ""


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _extract_json_str(text: str) -> str:
    """Extract a JSON object string from text that may contain extra content."""
    if not text:
        return ""
    text = text.strip()
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right > left:
        return text[left : right + 1]
    return text


def _parse_judge_response(raw: str) -> dict[str, Any]:
    """Parse a structured judge JSON response into {judgment, confidence, reasoning}.

    Falls back to plain-text ``correct``/``incorrect`` parsing if JSON fails.
    """
    default: dict[str, Any] = {
        "judgment": "unknown",
        "confidence": 0.0,
        "reasoning": "parse_failed",
    }
    if not raw:
        return default
    try:
        json_str = _extract_json_str(raw)
        parsed = json.loads(json_str)
        judgment = parsed.get("judgment", "unknown")
        if judgment not in ("correct", "incorrect", "partially_correct", "uncertain"):
            judgment = "unknown"
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        return {
            "judgment": judgment,
            "confidence": confidence,
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception:
        # Fallback: plain-text correct/incorrect detection
        text = raw.strip().lower()
        if text.startswith("correct") and not text.startswith("incorrect"):
            return {"judgment": "correct", "confidence": 0.8, "reasoning": f"plain_text:{raw[:200]}"}
        if text.startswith("incorrect"):
            return {"judgment": "incorrect", "confidence": 0.8, "reasoning": f"plain_text:{raw[:200]}"}
        return default


# ---------------------------------------------------------------------------
# Internal: call a single judge tier
# ---------------------------------------------------------------------------


def _call_tier(
    profile_name: str,
    prompt: str,
    api_stats: list | None = None,
) -> tuple[dict[str, Any], str]:
    """Call a single LLM tier for judge evaluation.

    Returns ``(parsed_result, raw_content)``.
    """
    client = _get_client(profile_name)
    raw = _llm_chat(client, prompt, temperature=0.0)

    if api_stats is not None:
        api_stats.append({
            "profile": profile_name,
            "model": client.default_model,
            "success": raw is not None,
        })

    if raw is None:
        return (
            {"judgment": "unknown", "confidence": 0.0, "reasoning": "call_failed"},
            "",
        )

    parsed = _parse_judge_response(raw)
    return parsed, raw


# ---------------------------------------------------------------------------
# Core: cascade answer judge
# ---------------------------------------------------------------------------


def cascade_answer_judge(
    question: str,
    ground_truth: str,
    prediction: str,
    primary_profile: str | None = None,
    secondary_profile: str | None = None,
    api_stats: list | None = None,
) -> tuple[float, str]:
    """Two-tier cascade judge for answer correctness.

    Flow:
      1. Exact (normalized) text match -> 1.0
      2. Primary model (config: reward_judge_primary_profile) -> high confidence -> done
      3. Secondary model (config: reward_judge_secondary_profile) -> final decision
      4. All tiers failed -> best-guess from earlier tiers or text match fallback

    Args:
        question: The original question text.
        ground_truth: The expected correct answer.
        prediction: The model's predicted answer.
        primary_profile: Override for the primary judge profile name.
        secondary_profile: Override for the secondary judge profile name.
        api_stats: Optional list to collect API call statistics.

    Returns:
        ``(score, explanation)`` where score is 0.0 (incorrect) or 1.0 (correct).
    """
    if not prediction:
        return 0.0, "missing_prediction"

    gt = str(ground_truth or "").strip()
    if not gt:
        return 0.0, "missing_ground_truth"

    # Resolve profiles from config
    cfg_primary, cfg_secondary = judge_profiles()
    if primary_profile is None:
        primary_profile = cfg_primary
    if secondary_profile is None:
        secondary_profile = cfg_secondary

    # ---- Step 0: Exact match ----
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(gt)
    if pred_norm and gt_norm and pred_norm == gt_norm:
        # print(
        #     f"[cascade_judge] ground_truth={gt!r} | model_answer={prediction!r} | "
        #     f"judgment=exact_match | confidence=1.0",
        #     flush=True,
        # )
        return 1.0, "exact_match"

    # Build the structured evaluation prompt
    judge_prompt = JUDGE_PROMPT.format(
        question=question[:2000],
        answer=gt[-2000:],
        prediction=prediction[-2000:],
    )

    chain_log: list[str] = []

    # ---- Step 1: Primary model ----
    primary_result: dict[str, Any] = {"judgment": "unknown", "confidence": 0.0}
    try:
        primary_result, primary_raw = _call_tier(
            primary_profile, judge_prompt, api_stats=api_stats,
        )
        pj = primary_result["judgment"]
        pc = primary_result["confidence"]
        # print(
        #     f"[cascade_judge] ground_truth={gt!r} | model_answer={prediction!r} | "
        #     f"primary_judgment={pj} | primary_confidence={pc:.3f} | "
        #     f"primary_model={primary_profile}",
        #     flush=True,
        # )
        chain_log.append(f"primary({primary_profile}:{pj},conf={pc:.3f})")

        if pj in ("correct", "incorrect") and pc >= CONFIDENCE_THRESHOLD:
            score = 1.0 if pj == "correct" else 0.0
            return score, "|".join(chain_log) + f"|{primary_raw}"
    except Exception as exc:
        print(
            f"[cascade_judge] primary profile '{primary_profile}' failed: {exc}",
            flush=True,
        )
        chain_log.append(f"primary({primary_profile}:FAILED)")

    # ---- Step 2: Secondary model (final decision) ----
    secondary_result: dict[str, Any] = {"judgment": "unknown", "confidence": 0.0}
    if secondary_profile:
        try:
            secondary_result, secondary_raw = _call_tier(
                secondary_profile, judge_prompt, api_stats=api_stats,
            )
            sj = secondary_result["judgment"]
            sc = secondary_result["confidence"]
            # print(
            #     f"[cascade_judge] -> secondary: judgment={sj} | confidence={sc:.3f} | "
            #     f"model={secondary_profile}",
            #     flush=True,
            # )
            chain_log.append(f"secondary({secondary_profile}:{sj},conf={sc:.3f})")

            if sj == "correct":
                return 1.0, "|".join(chain_log) + f"|{secondary_raw}"
            elif sj == "incorrect":
                return 0.0, "|".join(chain_log) + f"|{secondary_raw}"
            elif sj == "partially_correct":
                chain_log.append("partially_correct->incorrect")
                # print(
                #     "[cascade_judge] -> partially_correct treated as incorrect (binary mode)",
                #     flush=True,
                # )
                return 0.0, "|".join(chain_log) + f"|{secondary_raw}"
        except Exception as exc:
            print(
                f"[cascade_judge] secondary profile '{secondary_profile}' failed: {exc}",
                flush=True,
            )
            chain_log.append(f"secondary({secondary_profile}:FAILED)")
    else:
        chain_log.append("secondary(N/A)")

    # ---- Step 3: Fallback -- use best available information ----

    # 3a. If any tier said partially_correct, treat as incorrect
    for tier_result, tier_name in [
        (secondary_result, "secondary"),
        (primary_result, "primary"),
    ]:
        if tier_result["judgment"] == "partially_correct":
            chain_log.append(f"fallback({tier_name}_partially_correct->incorrect)")
            # print(
            #     f"[cascade_judge] -> fallback: {tier_name} said partially_correct, "
            #     f"treated as incorrect",
            #     flush=True,
            # )
            return 0.0, "|".join(chain_log)

    # 3b. Trust any low-confidence correct/incorrect from earlier tiers
    for tier_result, tier_name in [
        (secondary_result, "secondary"),
        (primary_result, "primary"),
    ]:
        if tier_result["judgment"] == "correct":
            chain_log.append(f"fallback(best_guess={tier_name}_correct)")
            # print(
            #     f"[cascade_judge] -> fallback: using {tier_name} low-conf correct",
            #     flush=True,
            # )
            return 1.0, "|".join(chain_log)
        if tier_result["judgment"] == "incorrect":
            chain_log.append(f"fallback(best_guess={tier_name}_incorrect)")
            # print(
            #     f"[cascade_judge] -> fallback: using {tier_name} low-conf incorrect",
            #     flush=True,
            # )
            return 0.0, "|".join(chain_log)

    # 3c. Final fallback: normalized text match
    fallback = 1.0 if pred_norm and gt_norm and pred_norm == gt_norm else 0.0
    chain_log.append(f"norm_match({fallback})")
    # print(
    #     f"[cascade_judge] -> all tiers exhausted, norm_match={fallback}", flush=True
    # )
    return fallback, "|".join(chain_log)
