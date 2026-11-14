import json
import re
from typing import Dict, List


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False).strip()
    except TypeError:
        return str(value).strip()


def _skill_name_from_raw(name: str) -> str:
    lowered = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unnamed_skill"


def _looks_like_code(text: str) -> bool:
    if not text:
        return False
    probes = (
        "def ",
        "class ",
        "import ",
        "from ",
        "return ",
        "if __name__",
        "```python",
    )
    lowered = text.lower()
    return any(p in lowered for p in probes)


def normalize_skill(skill: Dict) -> Dict:
    """Normalize legacy/new skill structures into the unified schema.

    Canonical fields (ONLY these are kept in the output):
      - skill_name: identifier (snake_case)
      - use_when: when to activate this skill
      - workflow: textual knowledge / workflow steps
      - code: optional Python code

    A valid skill should contain at least one non-empty field among
    workflow and code.
    """
    if not isinstance(skill, dict):
        return {}

    # ---- Map new canonical fields ----
    workflow = _to_text(skill.get("workflow", ""))
    code = _to_text(skill.get("code", ""))
    use_when = _to_text(skill.get("use_when", ""))

    # ---- Ingest old field names ----
    legacy_knowledge = _to_text(skill.get("knowledge", ""))
    legacy_description = _to_text(skill.get("description", ""))
    legacy_content = _to_text(skill.get("content", ""))
    legacy_type = str(skill.get("type", "")).strip().lower()

    if legacy_knowledge and not workflow:
        workflow = legacy_knowledge

    if legacy_description and not use_when:
        use_when = legacy_description

    # Old {type, content} format
    if legacy_content:
        if legacy_type == "code":
            if not code:
                code = legacy_content
        elif legacy_type == "knowledge":
            if not workflow:
                workflow = legacy_content
        else:
            if _looks_like_code(legacy_content):
                if not code:
                    code = legacy_content
            else:
                if not workflow:
                    workflow = legacy_content

    # ---- skill_name ----
    skill_name = _to_text(skill.get("skill_name", ""))
    legacy_name = _to_text(skill.get("name", ""))
    if not skill_name and legacy_name:
        skill_name = _skill_name_from_raw(legacy_name)
    skill_name = skill_name or _skill_name_from_raw(legacy_name or "unnamed_skill")

    # ---- Build clean output with ONLY canonical fields + metadata ----
    out: Dict = {
        "skill_name": skill_name,
        "use_when": use_when,
        "workflow": workflow,
        "code": code,
    }

    # Preserve metadata fields (not schema fields)
    for meta_key in ("skill_id", "usage_count", "success_count", "version",
                     "created_at", "evolved_from", "evolved_reason", "evolved_at",
                     "action", "builtin", "need_reason", "from_correct",
                     "source_question"):
        if meta_key in skill:
            out[meta_key] = skill[meta_key]

    return out


def skill_workflow(skill: Dict) -> str:
    """Return the workflow text of a skill."""
    return normalize_skill(skill).get("workflow", "")


# Backward compat alias — callers should migrate to skill_workflow
skill_knowledge = skill_workflow


def skill_use_when(skill: Dict) -> str:
    """Return the use_when text of a skill."""
    return normalize_skill(skill).get("use_when", "")


def skill_code(skill: Dict) -> str:
    return normalize_skill(skill).get("code", "")


def skill_has_payload(skill: Dict) -> bool:
    s = normalize_skill(skill)
    return bool(s.get("workflow") or s.get("code"))


def skill_components(skill: Dict) -> List[str]:
    s = normalize_skill(skill)
    parts: List[str] = []
    if s.get("workflow"):
        parts.append("workflow")
    if s.get("code"):
        parts.append("code")
    return parts


def skill_text_for_judge(skill: Dict) -> str:
    """Flatten skill payload into a single text block for judge/scoring prompts."""
    s = normalize_skill(skill)
    parts = []
    if s.get("workflow"):
        parts.append(f"Workflow:\n{s['workflow']}")
    if s.get("code"):
        parts.append(f"Code:\n{s['code']}")
    return "\n\n".join(parts).strip()
