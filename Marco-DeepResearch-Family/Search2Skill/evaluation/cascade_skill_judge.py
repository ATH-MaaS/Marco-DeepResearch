"""Cascade skill quality judge.

Two-tier cascade for skill quality scoring (informativeness / groundedness / code_quality):
  1. Primary model (config: rl.skill_judge_primary_profile, default "qwen3-30b-a3b-local")
     → if confidence >= threshold → use its score directly
  2. Fallback model (config: rl.skill_judge_fallback_profile, default "qwen36-plus")
     → final decision

Configuration via config.yaml ``rl`` section:
  - skill_judge_primary_profile   (default: "qwen3-30b-a3b-local")
  - skill_judge_fallback_profile  (default: "qwen36-plus")

Env var overrides:
  EXSKILL_SKILL_JUDGE_PRIMARY_PROFILE
  EXSKILL_SKILL_JUDGE_FALLBACK_PROFILE
  EXSKILL_SKILL_JUDGE_CONFIDENCE_THRESHOLD  (default: 0.95)
"""

import json
import os
import re
from typing import Any

from .cascade_judge import load_config, sync_chat

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = float(
    os.getenv("EXSKILL_SKILL_JUDGE_CONFIDENCE_THRESHOLD", "0.96")
)


def _skill_judge_profiles() -> tuple[str, str]:
    """Return (primary_profile, fallback_profile) for cascade skill judge."""
    cfg = load_config()
    rl_cfg = cfg.get("rl", {})
    primary = os.environ.get(
        "EXSKILL_SKILL_JUDGE_PRIMARY_PROFILE",
        rl_cfg.get("skill_judge_primary_profile", "qwen3-30b-a3b-local"),
    )
    fallback = os.environ.get(
        "EXSKILL_SKILL_JUDGE_FALLBACK_PROFILE",
        rl_cfg.get("skill_judge_fallback_profile", "qwen36-plus"),
    )
    return primary, fallback


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Extract JSON object from raw LLM response."""
    if not raw:
        return {}
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


def _get_confidence(parsed: dict[str, Any]) -> float:
    """Extract confidence from parsed JSON, default 0.5."""
    try:
        return max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    except (TypeError, ValueError):
        return 0.5


# ---------------------------------------------------------------------------
# Core cascade call
# ---------------------------------------------------------------------------


def cascade_skill_chat(
    prompt: str,
    api_stats: list | None = None,
) -> str:
    """Call primary model first; if confidence < threshold, fallback to secondary.

    Returns the raw response string from whichever model is used.
    """
    primary, fallback = _skill_judge_profiles()

    # --- Tier 1: Primary model ---
    primary_raw = sync_chat(primary, prompt, temperature=0.0, api_stats=api_stats)
    if primary_raw:
        parsed = _parse_json_response(primary_raw)
        conf = _get_confidence(parsed)
        if conf >= CONFIDENCE_THRESHOLD:
            return primary_raw

    # --- Tier 2: Fallback model ---
    fallback_raw = sync_chat(fallback, prompt, temperature=0.0, api_stats=api_stats)
    return fallback_raw or primary_raw or ""
