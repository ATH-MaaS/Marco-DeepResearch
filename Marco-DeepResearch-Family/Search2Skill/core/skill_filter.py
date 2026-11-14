"""General-purpose hardcoded skill filter.

Detects skills that memorize instance-level data (specific answers, lookup tables)
rather than encoding generalizable procedures. Domain-agnostic.

Usage:
    from core.skill_filter import is_hardcoded, filter_skill_library

    # Single skill check
    flagged, reason = is_hardcoded(skill_dict)

    # Batch filter
    clean_skills = filter_skill_library(skills_list)
"""

import re
from typing import Dict, List, Tuple


def is_hardcoded(skill: Dict) -> Tuple[bool, str]:
    """Detect whether a skill encodes instance data rather than a generalizable method.

    Core principle: a useful skill teaches HOW to solve a class of problems (procedure).
    A hardcoded skill stores WHAT the answer is for specific inputs (memorization).

    Returns:
        (is_hardcoded, reason) — reason is empty string if not hardcoded.
    """
    code = skill.get("code", "") or ""
    workflow = skill.get("workflow", "") or ""

    # ─── Rule 1: Trivial function ─────────────────────────────────────────
    # Function body is a single `return <literal>`.
    # A skill should compute something, not just hold a value.
    reason = _check_trivial_function(code)
    if reason:
        return True, reason

    # ─── Rule 2: Lookup table in code ─────────────────────────────────────
    # Dict with many long string keys = memorized input->output mapping.
    reason = _check_lookup_table(code)
    if reason:
        return True, reason

    # ─── Rule 3: Instance memorization in workflow ────────────────────────
    # Workflow stores specific data instances (long opaque tokens) instead of steps.
    reason = _check_instance_memorization(workflow)
    if reason:
        return True, reason

    # ─── Rule 4: Overfitted language ─────────────────────────────────────
    # Workflow describes solving ONE specific problem, not a class.
    reason = _check_overfitted_language(workflow)
    if reason:
        return True, reason

    # ─── Rule 5: High constant-to-logic ratio ────────────────────────────
    # Many magic numbers with little branching = hardcoded lookup, not algorithm.
    reason = _check_constant_density(code)
    if reason:
        return True, reason

    return False, ""


def filter_skill_library(skills: List[Dict]) -> List[Dict]:
    """Filter a skill library, removing hardcoded skills.

    Returns:
        List of skills that passed the filter.
    """
    return [s for s in skills if not is_hardcoded(s)[0]]


def filter_skill_library_verbose(skills: List[Dict]) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    """Filter with details on what was removed.

    Returns:
        (kept_skills, removed_list) where removed_list is [(skill_name, reason), ...]
    """
    kept = []
    removed = []
    for s in skills:
        flagged, reason = is_hardcoded(s)
        if flagged:
            removed.append((s.get("skill_name", "unknown"), reason))
        else:
            kept.append(s)
    return kept, removed


# ═══════════════════════════════════════════════════════════════════════════
# Internal check functions
# ═══════════════════════════════════════════════════════════════════════════


_TRIVIAL_RETURNS = frozenset([
    "True", "False", "None", "0", "1", "0.0", "1.0",
    '""', "''", "[]", "{}", "0.5",
])


def _check_trivial_function(code: str) -> str:
    """Detect functions whose entire body is `return <constant>`."""
    func_bodies = re.findall(r"def\s+\w+\([^)]*\):\s*\n((?:[ \t]+.*\n)*)", code)
    for body in func_bodies:
        meaningful = [
            l.strip() for l in body.split("\n")
            if l.strip()
            and not l.strip().startswith("#")
            and not l.strip().startswith('"""')
            and not l.strip().startswith("'''")
        ]
        if len(meaningful) != 1:
            continue
        line = meaningful[0]
        if not re.match(r"return\s+[^()\[\]{}]+$", line):
            continue
        ret_val = line.replace("return", "", 1).strip()
        if ret_val in _TRIVIAL_RETURNS:
            continue
        # Is it a bare literal (number or string)?
        if re.match(r"^[\d\.\-]+$", ret_val) or re.match(r'^["\'].*["\']$', ret_val):
            return f"trivial_function(return {ret_val})"
    return ""


def _check_lookup_table(code: str, min_entries: int = 5, min_key_len: int = 12) -> str:
    """Detect code with large hardcoded string-key dicts."""
    long_key_entries = re.findall(r"""[\"']([^\"'\n]{""" + str(min_key_len) + r""",})[\"']\s*:""", code)
    # Exclude URLs, file paths, natural language (has spaces)
    data_keys = [
        k for k in long_key_entries
        if not k.startswith("http")
        and not k.startswith("/")
        and " " not in k
    ]
    if len(data_keys) >= min_entries:
        return f"lookup_table_in_code({len(data_keys)} entries)"
    return ""


def _check_instance_memorization(
    workflow: str, min_opaque_tokens: int = 4, min_token_len: int = 25
) -> str:
    """Detect workflows that store specific data instances as opaque tokens."""
    opaque_tokens = re.findall(
        r"(?<!\w)[A-Za-z0-9@\[\]()=/#\\+\-.:_]{" + str(min_token_len) + r",}(?!\w)",
        workflow,
    )
    # Filter out URLs, paths, simple identifiers
    filtered = [
        t for t in opaque_tokens
        if not t.startswith("http")
        and not t.startswith("/")
        and not t.startswith("www.")
        and not re.match(r"^[a-z_]+$", t)  # snake_case identifiers are fine
    ]
    if len(filtered) >= min_opaque_tokens:
        return f"instance_memorization_in_workflow({len(filtered)} opaque tokens)"
    return ""


_OVERFIT_PATTERNS = [
    r"(?:for|in)\s+this\s+(?:specific|particular)\s+(?:reaction|problem|case|question|molecule|example|input|scenario)",
    r"experimental\s+data\s+suggests\s+(?:a\s+)?(?:yield|value|result)",
    r"the\s+(?:correct|right|exact)\s+answer\s+(?:is|would be|=)",
    r"(?:the|this)\s+(?:specific|exact)\s+(?:value|result|answer|yield|output)\s+is\s+\d",
]


def _check_overfitted_language(workflow: str) -> str:
    """Detect workflow text that describes solving one specific instance."""
    for pat in _OVERFIT_PATTERNS:
        if re.search(pat, workflow, re.I):
            return "overfitted_language"
    return ""



def _check_constant_density(code: str, min_magic: int = 6, max_logic: int = 1) -> str:
    """Detect code with many magic numbers but little control flow."""
    if not code.strip():
        return ""
    # Find specific floats (2+ decimal places, not common constants)
    specific_floats = re.findall(r"(?<![a-zA-Z_])\d+\.\d{2,}", code)
    common = {0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 0.01, 0.001, 0.1}
    magic_numbers = [f for f in specific_floats if float(f) not in common]
    logic_keywords = len(re.findall(r"\b(?:if|elif|else|for|while|try)\b", code))
    if len(magic_numbers) >= min_magic and logic_keywords <= max_logic:
        return f"high_constant_density({len(magic_numbers)} magic numbers, {logic_keywords} branches)"
    return ""
