"""
Local HTS Tree Tools — 本地税则树查询工具注册到 hermes-agent.

将 run_test.py 中的 3 个 HTS 工具（get_hs_subcategories/get_hs_code_info/validate_hs_code）
注册到 hermes-agent registry（toolset="hts"），供 batch_test.py 使用。

使用方式：
    from custom_hts_tools import register_hts_tools
    register_hts_tools()  # 注册本地 HTS 工具到 hermes-agent registry
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Path Configuration
# ═══════════════════════════════════════════════════════════════

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
HTS_TREE_PATH = DATA_DIR / "hts_tree.json"
HTS_RAW_PATH = DATA_DIR / "hts_2025_revision_15.json"

# ═══════════════════════════════════════════════════════════════
# HTS Tree Data (module-level singletons)
# ═══════════════════════════════════════════════════════════════

_hts_tree: Dict[str, List[Dict]] = {}
_hts_codes: Dict[str, Dict] = {}
_children_index: Dict[str, List[Dict]] = {}
_all_known_codes: Dict[str, str] = {}


def load_hts_tree() -> None:
    """Load HTS tree data and build complete parent→children index."""
    global _hts_tree, _hts_codes, _children_index, _all_known_codes
    if _hts_tree:
        return  # Already loaded

    with open(HTS_TREE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    _hts_tree = data["tree"]
    _hts_codes = data["codes"]

    # Build COMPLETE children index covering all levels (2→4→6→8→10)
    _children_index.update(_hts_tree)

    # Collect ALL known codes from tree + codes + raw
    all_known_codes: Dict[str, str] = {}

    for key in _hts_tree:
        info = _hts_codes.get(key, {})
        all_known_codes[key] = info.get("desc", "")

    for parent, children in _hts_tree.items():
        for ch in children:
            code = ch["hs_code"]
            if code not in all_known_codes or not all_known_codes[code]:
                all_known_codes[code] = ch.get("desc", "")

    for code, info in _hts_codes.items():
        if code not in all_known_codes or not all_known_codes[code]:
            all_known_codes[code] = info.get("desc", "")

    # Enrich from raw USITC data
    if HTS_RAW_PATH.exists():
        with open(HTS_RAW_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        for record in raw_data:
            htsno = (record.get("htsno") or "").replace(".", "").replace(" ", "")
            desc = record.get("description", "")
            if htsno and desc:
                if htsno not in all_known_codes or not all_known_codes[htsno]:
                    all_known_codes[htsno] = desc

    # Register ALL intermediate prefixes
    all_codes_list = list(all_known_codes.keys())
    for code in all_codes_list:
        for prefix_len in [2, 4, 6, 8]:
            if prefix_len < len(code):
                prefix = code[:prefix_len]
                if prefix not in all_known_codes:
                    all_known_codes[prefix] = ""

    # For each level, ensure parent→child mappings exist
    for target_len in [4, 6, 8, 10]:
        parent_len = target_len - 2
        codes_at_level = [c for c in all_known_codes if len(c) == target_len]
        for code in codes_at_level:
            parent = code[:parent_len]
            _children_index.setdefault(parent, [])
            existing = {c["hs_code"] for c in _children_index[parent]}
            if code not in existing:
                desc = all_known_codes.get(code, "")
                is_leaf = code not in _hts_tree and target_len == 10
                if target_len == 4:
                    formatted = code
                elif target_len == 6:
                    formatted = f"{code[:4]}.{code[4:6]}"
                elif target_len == 8:
                    formatted = f"{code[:4]}.{code[4:6]}.{code[6:8]}"
                else:
                    formatted = f"{code[:4]}.{code[4:6]}.{code[6:8]}.{code[8:10]}"
                _children_index[parent].append({
                    "hs_code": code,
                    "hs_code_formatted": formatted,
                    "desc": desc,
                    "is_leaf": is_leaf,
                })

    # Post-process: fill empty descriptions
    for parent, children in list(_children_index.items()):
        for ch in children:
            if not ch.get("desc"):
                code = ch["hs_code"]
                sub_children = _children_index.get(code, [])
                if len(sub_children) == 1:
                    ch["desc"] = sub_children[0].get("desc", "")
                elif sub_children:
                    descs = [c.get("desc", "") for c in sub_children if c.get("desc")]
                    if descs:
                        ch["desc"] = descs[0] + (" (and others)" if len(descs) > 1 else "")

    # Sync back
    for parent, children in _children_index.items():
        for ch in children:
            code = ch["hs_code"]
            desc = ch.get("desc", "")
            if desc and (code not in all_known_codes or not all_known_codes[code]):
                all_known_codes[code] = desc
    _all_known_codes.update(all_known_codes)

    logger.info(
        "HTS tree loaded: %d tree nodes, %d codes, %d indexed parents",
        len(_hts_tree), len(_hts_codes), len(_children_index)
    )


# ═══════════════════════════════════════════════════════════════
# Tool Functions
# ═══════════════════════════════════════════════════════════════


def get_hs_subcategories(parent_code: str) -> str:
    """Get child categories for a given parent HS code."""
    load_hts_tree()
    parent_code = re.sub(r"[^0-9]", "", str(parent_code).strip())

    children = _children_index.get(parent_code, [])
    if children:
        lines = []
        for ch in sorted(children, key=lambda x: x["hs_code"]):
            code = ch["hs_code"]
            desc = ch["desc"]
            leaf = " [LEAF]" if ch.get("is_leaf") else ""
            code_info = _hts_codes.get(code, {})
            duty = ""
            if code_info.get("general_duty"):
                duty = f" | Duty: {code_info['general_duty']}"
            lines.append(f"  {code} - {desc}{leaf}{duty}")
        return f"Children of {parent_code} ({len(children)} items):\n" + "\n".join(lines)

    plen = len(parent_code)
    child_len = {2: 4, 4: 6, 6: 8, 8: 10}.get(plen)
    if not child_len:
        return f"No subcategories found for '{parent_code}'. Valid parent code lengths: 2, 4, 6, or 8 digits."

    found = []
    for code, info in _hts_codes.items():
        if code.startswith(parent_code) and len(code) == child_len:
            duty = f" | Duty: {info['general_duty']}" if info.get("general_duty") else ""
            found.append(f"  {code} - {info['desc']}{duty}")

    if found:
        return f"Children of {parent_code} ({len(found)} items):\n" + "\n".join(sorted(found))
    return f"No subcategories found for '{parent_code}'."


def get_hs_code_info(hs_code: str) -> str:
    """Get detailed information for a specific HS code."""
    load_hts_tree()
    hs_code = re.sub(r"[^0-9]", "", str(hs_code).strip())
    info = _hts_codes.get(hs_code)
    if info:
        parts = [
            f"Code: {hs_code} ({info.get('formatted', hs_code)})",
            f"Description: {info['desc']}",
        ]
        if info.get("general_duty"):
            parts.append(f"General Duty: {info['general_duty']}")
        if info.get("special_duty"):
            parts.append(f"Special Duty: {info['special_duty']}")
        chain = []
        for n in [2, 4, 6, 8]:
            if n < len(hs_code):
                prefix = hs_code[:n]
                p_info = _hts_codes.get(prefix)
                if p_info:
                    chain.append(f"  {prefix}: {p_info['desc']}")
        if chain:
            parts.append("Classification path:\n" + "\n".join(chain))
        return "\n".join(parts)

    if hs_code in _all_known_codes:
        desc = _all_known_codes[hs_code]
        parts = [f"Code: {hs_code}"]
        if desc:
            parts.append(f"Description: {desc}")
        sub_children = _children_index.get(hs_code, [])
        if sub_children:
            parts.append(f"Has {len(sub_children)} subcategories — use get_hs_subcategories('{hs_code}') to see them.")
        else:
            parts.append("Type: LEAF (10-digit statistical suffix)")
        return "\n".join(parts)

    return f"Code '{hs_code}' not found in HTS database."


def validate_hs_code(hs_code: str) -> str:
    """Validate whether a 10-digit HS code exists in the HTS database."""
    load_hts_tree()
    hs_code = re.sub(r"[^0-9]", "", str(hs_code).strip())
    if len(hs_code) != 10:
        return f"Invalid: '{hs_code}' is not 10 digits (got {len(hs_code)})."
    if hs_code in _hts_codes:
        info = _hts_codes[hs_code]
        return f"VALID: {hs_code} ({info.get('formatted','')}) - {info['desc']}"
    for n in [8, 6, 4, 2]:
        prefix = hs_code[:n]
        if prefix in _hts_codes:
            return (
                f"INVALID: {hs_code} does not exist. "
                f"Nearest valid prefix: {prefix} ({_hts_codes[prefix]['desc']}). "
                f"Use get_hs_subcategories('{prefix}') to see valid children."
            )
    return f"INVALID: {hs_code} does not exist in HTS database."


# ═══════════════════════════════════════════════════════════════
# Tool Schemas (hermes-agent registry format)
# ═══════════════════════════════════════════════════════════════

HTS_SUBCATEGORIES_SCHEMA = {
    "name": "get_hs_subcategories",
    "description": (
        "Get child HS code categories for a given parent code. "
        "Use this to drill down the HTS tree: 2-digit->4-digit->6-digit->8-digit->10-digit. "
        "Start with a 2-digit chapter code and progressively narrow down."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "parent_code": {
                "type": "string",
                "description": "Parent HS code (2/4/6/8 digits). E.g., '71' for chapter 71, '7117' for heading 7117.",
            }
        },
        "required": ["parent_code"],
    },
}

HTS_CODE_INFO_SCHEMA = {
    "name": "get_hs_code_info",
    "description": (
        "Get detailed information for a specific HS code including description, "
        "duty rates, and classification path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "hs_code": {
                "type": "string",
                "description": "HS code to look up (any length: 2-10 digits).",
            }
        },
        "required": ["hs_code"],
    },
}

HTS_VALIDATE_SCHEMA = {
    "name": "validate_hs_code",
    "description": (
        "Validate whether a 10-digit HS code exists in the US HTS database. "
        "Use this to verify your final answer before submitting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "hs_code": {
                "type": "string",
                "description": "10-digit HS code to validate.",
            }
        },
        "required": ["hs_code"],
    },
}


# ═══════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════


def _check_hts_available() -> bool:
    """HTS tools are available if data files exist."""
    return HTS_TREE_PATH.exists()


def register_hts_tools():
    """注册本地 HTS 树查询工具到 hermes-agent registry (toolset='hts')."""
    from tools.registry import registry

    # Ensure tree is loaded
    load_hts_tree()

    registry.register(
        name="get_hs_subcategories",
        toolset="hts",
        schema=HTS_SUBCATEGORIES_SCHEMA,
        handler=lambda args, **kw: get_hs_subcategories(args.get("parent_code", "")),
        check_fn=_check_hts_available,
        emoji="🌳",
        max_result_size_chars=100_000,
        override=True,
    )

    registry.register(
        name="get_hs_code_info",
        toolset="hts",
        schema=HTS_CODE_INFO_SCHEMA,
        handler=lambda args, **kw: get_hs_code_info(args.get("hs_code", "")),
        check_fn=_check_hts_available,
        emoji="📋",
        max_result_size_chars=50_000,
        override=True,
    )

    registry.register(
        name="validate_hs_code",
        toolset="hts",
        schema=HTS_VALIDATE_SCHEMA,
        handler=lambda args, **kw: validate_hs_code(args.get("hs_code", "")),
        check_fn=_check_hts_available,
        emoji="✅",
        max_result_size_chars=10_000,
        override=True,
    )

    logger.info("✅ HTS tree tools registered (toolset='hts'): get_hs_subcategories, get_hs_code_info, validate_hs_code")
