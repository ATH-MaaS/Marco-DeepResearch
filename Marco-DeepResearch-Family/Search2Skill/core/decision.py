import json
import re
from textwrap import dedent
from typing import Dict, List, Optional, Union

from json_repair import repair_json


# ==================== Prompt Templates ====================

#   - DEFAULT TO EXPLORE: For any problem involving mathematical theorems, specialized techniques, non-trivial algorithms, or domain-specific formulas, you should use <EXPLORE> as your FIRST action — even if you think you might know the answer. Exploration builds reusable skills and catches subtle errors that direct solving misses. Only skip exploration for truly elementary problems (basic arithmetic, simple algebra).

EXECUTION_PROMPT = dedent(
    """\
    You are an intelligent assistant solving problems in multiple rounds.

    ## Available Skills:
    {formatted_skills}

    ## Question:
    {question}
    {context_section}

    Rules:
    - Think step-by-step in plain text, then output EXACTLY ONE action tag at the end.
    - You will receive feedback after each action.
    - Never output raw code directly; always use <CODE> to run Python code.
    - CRITICAL: Do NOT guess when uncertain. If you are not highly confident (>90%) in your answer, you MUST use <EXPLORE> to search for evidence first. Getting the answer wrong is worse than spending an extra round exploring.
 
    Uncertainty signals (any of these means you MUST EXPLORE):
    - The question asks about specific facts, names, dates, numbers, or formulas you are not 100% sure about
    - You do not know or are not fully sure about a concept, knowledge point, procedure, workflow, API, library usage, command, syntax, or implementation detail needed to solve the task
    - You find yourself torn between two or more options
    - You are relying on vague memory rather than solid knowledge
    - The problem requires a specific algorithm, technique, or mathematical method that you are not fully confident you can implement correctly (e.g., a DP variant, number-theoretic formula, geometric algorithm, or specialized data structure)
    - The problem involves competition-level mathematics, olympiad problems, or advanced undergraduate+ topics — these almost always benefit from exploration

    Actions (Choose exactly one):
    1) Activate a skill: <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>. Once you have activated a skill, you can view the knowledge contained within it, or directly utilize the functions included in the skill within <CODE>.
    2) Run Python code: <CODE>
       your python code here
       </CODE>
    3) Explore — USE THIS when uncertain or missing knowledge: <EXPLORE>what specific information, knowledge, method, example, workflow, or implementation detail you need to verify</EXPLORE>
    4) Answer — ONLY when confident: <ANSWER>final answer (or option letter like B for multiple choice)</ANSWER>
    """
).strip()

EXECUTION_PROMPT_NO_CODE = dedent(
    """\
    You are an intelligent assistant solving problems in multiple rounds.

    ## Available Skills:
    {formatted_skills}

    ## Question:
    {question}
    {context_section}

    Rules:
    - Think step-by-step in plain text, then output EXACTLY ONE action tag at the end.
    - You will receive feedback after each action.
    - CRITICAL: Do NOT guess when uncertain. If you are not highly confident (>90%) in your answer, you MUST use <EXPLORE> to search for evidence first. Getting the answer wrong is worse than spending an extra round exploring.
    - Tip: When in doubt, prefer <EXPLORE> over guessing — gathering evidence first usually leads to better answers.
    - DEFAULT TO EXPLORE: For any problem involving mathematical theorems, specialized techniques, non-trivial algorithms, or domain-specific formulas, you should use <EXPLORE> as your FIRST action — even if you think you might know the answer. Exploration builds reusable skills and catches subtle errors that direct solving misses. Only skip exploration for truly elementary problems (basic arithmetic, simple algebra).

    Uncertainty signals (any of these means you MUST EXPLORE):
    - The question asks about specific facts, names, dates, numbers, or formulas you are not 100% sure about
    - You find yourself torn between two or more options
    - The topic is highly specialized or obscure
    - You are relying on vague memory rather than solid knowledge
    - The problem requires a specific algorithm, technique, or mathematical method that you are not fully confident you can implement correctly (e.g., a DP variant, number-theoretic formula, geometric algorithm, or specialized data structure)
    - The problem involves competition-level mathematics, olympiad problems, or advanced undergraduate+ topics — these almost always benefit from exploration

    Actions (Choose exactly one):
    1) Explore — RECOMMENDED as your first action: <EXPLORE>what specific information you need to verify</EXPLORE>
    2) Activate a skill: <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>. Once you have activated a skill, you can view the knowledge contained within it.
    3) Answer — ONLY when confident: <ANSWER>final answer (or option letter like B for multiple choice)</ANSWER>
    """
).strip()

EXECUTION_PROMPT_EXPLORE_SEARCH = dedent(
    """\
    You are in the exploration phase. Goal: Gather evidence to answer the question. 
    Do not restate the question. Previous context is available.
    
    Current goal: {exploration_goal}
    
    Rules:
    - Think step-by-step in plain text, then output EXACTLY ONE action tag at the end.
    - You must generate a skill before answering or calling other skills.
    
    ## Search Rules — what to search for:
    - NEVER search for the problem itself or copy-paste the question into queries.
    - Instead, search for the **underlying knowledge** needed to solve it:
      * Formulas, theorems, or domain knowledge (e.g. "VSEPR theory bond angles")
      * Library/API usage (e.g. "python selfies library decode example", "rdkit MolFromSmiles tutorial")
      * Algorithms, techniques, or mathematical methods (e.g. "digit DP tutorial with modulo constraint", "monotonic stack range query algorithm", "Euler totient modular inverse", "how to convert SELFIES to SMILES python")
    - Your goal is to **learn the method or tool**, then apply it yourself.
    
    Recommended exploration chain (follow this order):
    Step 1. <QUERY> — Search for methods, formulas, or library/API usage relevant to the problem type.
    Step 2. <PAGE_VISIT> — Pick the most relevant URLs from search results and visit them to read full content. Search snippets are too short to be reliable; you MUST visit pages to get accurate, detailed information before generating a skill.
    Step 3. <GENERATE_SKILL> — After multiple rounds of <QUERY> and <PAGE_VISIT>, once you believe you have gathered sufficient information, you can synthesize the detailed page content into a reusable skill.
    
    CRITICAL: Do NOT skip PAGE_VISIT. Search snippets are brief summaries that often lack critical details or contain misleading fragments. Always visit at least one high-quality page to read the full content before generating a skill.
    
    Actions (Choose exactly one per round):
    1) Search web (at most 3 queries per call): <QUERY>["query 1", "query 2"]</QUERY>
    2) Visit pages to read full content (at most 4 URLs per call): <PAGE_VISIT>{{"urls": ["url1", "url2"], "goal": "what to extract"}}</PAGE_VISIT>
    3) Generate Skill (only after visiting pages):
    <GENERATE_SKILL>{{
        "skill_name": "name_with_underscores",
        "use_when": "when to activate this skill — describe the problem type or scenario",
        "workflow": "step-by-step workflow, distilled facts, formulas, constants from page content (or empty string)",
        "code": "reusable Python `def` functions (or empty string)"
    }}</GENERATE_SKILL>
    
    Skill Design Rules (<GENERATE_SKILL>):
    - Must contain "skill_name" and at least one non-empty "workflow" or "code".
    - Generalization Rule: Do not just solve the immediate question. Synthesize ALL uncovered domain knowledge into a broadly applicable skill for similar future problems. CRITICAL: You must extract non-obvious domain insights from the [Supplementary knowledge] sections in page visit results and include them.
    - "workflow" (Facts & Insights): Distilled facts, formulas, rules, and step-by-step procedures.
    - "code" (Computation & Logic): ONLY `def` functions (no top-level execution/prints). Loaded as a library. Use for complex math or external libs. ENCOURAGE parameterized functions: every function must accept inputs as arguments and compute results dynamically. FORBIDDEN returning a hardcoded constant, never write a function whose sole purpose is to return a fixed value (e.g., `def get_gravity(): return 9.8`). Such facts belong in "workflow", not in code.
    - "code" functions must be self-contained and not rely on any external fetching helpers. All reference data should be distilled into "workflow" instead.
    - Groundedness Rule: All factual claims, formulas, and parameters in "workflow" and "code" should be based on content you read via <PAGE_VISIT>. Do not substitute your internal knowledge for domain-specific facts. If a fact was not found in the visited pages, omit it rather than guess.
    
    Example — a skill combining workflow and computation:
    <GENERATE_SKILL>{{
        "skill_name": "radionuclide_decay_solver",
        "use_when": "Solving radioactive decay problems: look up half-lives and compute remaining quantities",
        "workflow": "Decay formula: N = N0 * (1/2)^(t / t_half). Common half-lives: C-14: 5730 years, U-238: 4.468e9 years, K-40: 1.248e9 years, Ra-226: 1600 years, Co-60: 5.27 years, I-131: 8.02 days, Sr-90: 28.8 years.",
        "code": "def decay(n0, half_life, t):\\n    return n0 * (0.5 ** (t / half_life))\\n\\ndef time_to_fraction(fraction, half_life):\\n    import math\\n    return half_life * math.log2(1 / fraction)"
    }}</GENERATE_SKILL>
    """
).strip()

TEACHER_SKILL_GENERATION_PROMPT = dedent(
    """\
    You are an expert knowledge distillation assistant. Your ONLY task is to generate a reusable skill based EXCLUSIVELY on the web search results and page content provided below.

    *** ABSOLUTE RULES — READ CAREFULLY ***
    1. You MUST NOT use any of your own prior knowledge, training data, or internal beliefs.
    2. You MUST ONLY extract and summarize information that is EXPLICITLY present in the provided search results and page content below.
    3. If the search results do not contain enough information about a topic, you MUST leave that part EMPTY rather than filling in from your own knowledge.
    4. Every single fact, formula, constant, or data point in your output MUST be directly traceable to the provided content. If you cannot point to where in the search results a piece of information came from, you MUST NOT include it.
    5. Do NOT infer, deduce, or extrapolate beyond what is explicitly stated in the search results.

    Exploration goal: {exploration_goal}

    ---BEGIN SEARCH RESULTS AND PAGE CONTENT---
    {search_context}
    ---END SEARCH RESULTS AND PAGE CONTENT---

    *** REMINDER: Your skill MUST contain ONLY information from the search results above. ***
    *** Do NOT include any knowledge from your training data, even if you believe it to be correct. ***

    Generate a skill in the following JSON format:
    {{{{
        "skill_name": "name_with_underscores",
        "use_when": "when to activate this skill — describe the problem type or scenario",
        "workflow": "step-by-step workflow, distilled facts, formulas, constants from the search results ONLY (or empty string if no reliable information found)",
        "code": "reusable Python `def` functions (or empty string)"
    }}}}

    Skill Design Rules:
    - Must contain "skill_name" and at least one non-empty "workflow" or "code".
    - Generalization Rule: Synthesize ALL uncovered domain knowledge from the search results into a broadly applicable skill.
    - "workflow": Distilled facts, formulas, and rules found EXPLICITLY in the search results. Do NOT add any fact not present above.
    - "code": ONLY `def` functions (no top-level execution/prints). FORBIDDEN returning hardcoded constants — such facts belong in "workflow".
    - FORBIDDEN: Including any information NOT present in the search results above.

    Output ONLY the JSON skill object, no other text.
    """
).strip()

EXECUTION_PROMPT_EXPLORE_SEARCH_NO_SKILL = dedent(
    """\
    You are in the exploration phase. Goal: Gather evidence to answer the question directly.
    Do not restate the question. Previous context is available.
    
    Current goal: {exploration_goal}
    
    Rules:
    - Think step-by-step in plain text, then output EXACTLY ONE action tag at the end.
    - After gathering enough information, answer the question directly with <ANSWER>.
    - Do NOT generate skills. Go straight from exploration to answering.
    
    Recommended exploration chain (follow this order):
    Step 1. <QUERY> — Use Google Search to find candidate pages that may contain the required knowledge.
    Step 2. <PAGE_VISIT> — Pick the most relevant URLs from search results and visit them to read full content. Search snippets are too short to be reliable; you MUST visit pages to get accurate, detailed information before answering.
    Step 3. <ANSWER> — After multiple rounds of <QUERY> and <PAGE_VISIT>, once you believe you have gathered sufficient information, you can answer based on the detailed page content you read.
    
    CRITICAL: Do NOT skip PAGE_VISIT. Search snippets are brief summaries that often lack critical details or contain misleading fragments. Always visit at least one high-quality page to read the full content before answering.
    
    Actions (Choose exactly one per round):
    1) Search web (at most 3 queries per call): <QUERY>["query 1", "query 2"]</QUERY>
    2) Visit pages to read full content (at most 4 URLs per call): <PAGE_VISIT>{{"urls": ["url1", "url2"], "goal": "what to extract"}}</PAGE_VISIT>
    3) Answer (only after visiting pages): <ANSWER>final answer (or option letter like B for multiple choice)</ANSWER>

 """
).strip()

EXECUTION_PROMPT_CODE_ONLY = dedent(
    """\
    You are an intelligent assistant solving problems with Python code. Exploration is disabled.

    ## Question:
    {question}
    {context_section}

    Rules:
    - Think step-by-step in plain text, then output EXACTLY ONE action tag at the end.
    - You will receive feedback after each action.
    - Never output raw code directly; always use <CODE> to run Python code.

    Actions (Choose exactly one):
    1) Run Python code: <CODE>
       your python code here
       </CODE>
    2) Answer: <ANSWER>final answer (or option letter like B for multiple choice)</ANSWER>
    """
).strip()

EXECUTION_PROMPT_INTERNAL = dedent(
    """\
    You are an intelligent assistant solving problems using skills. Exploration is disabled.

    ## Available Skills:
    {formatted_skills}

    ## Question:
    {question}
    {context_section}

    Rules:
    - Think step-by-step in plain text, then output EXACTLY ONE action tag at the end.
    - You will receive feedback after each action.
    - Never output raw code directly; always use <CODE> to run Python code.

    Actions (Choose exactly one):
    1) Activate a skill: <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>
    2) Run Python code: <CODE>
       your python code here
       </CODE>
    3) Answer: <ANSWER>final answer (or option letter like B for multiple choice)</ANSWER>
    """
).strip()

EXECUTION_PROMPT_INTERNAL_NO_CODE = dedent(
    """\
    You are an intelligent assistant solving problems using skills. Exploration is disabled.

    ## Available Skills:
    {formatted_skills}

    ## Question:
    {question}
    {context_section}

    Rules:
    - Think step-by-step in plain text, then output EXACTLY ONE action tag at the end.
    - You will receive feedback after each action.
    - IMPORTANT: Your final answer MUST be wrapped in <ANSWER></ANSWER> tags. For coding problems, put the complete solution code inside <ANSWER>...</ANSWER>. Do NOT output bare code blocks without the <ANSWER> tag.

    Actions (Choose exactly one):
    1) Activate a skill: <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>
    2) Answer: <ANSWER>final answer (or option letter like B for multiple choice)</ANSWER>
    """
).strip()

EXECUTION_PROMPT_ANSWER_ONLY = dedent(
    """\
    You are an intelligent assistant. Exploration is disabled.

    ## Question:
    {question}
    {context_section}

    Rules:
    - Think step-by-step in plain text, then output EXACTLY ONE action tag at the end.
    - IMPORTANT: Your final answer MUST be wrapped in <ANSWER></ANSWER> tags. For coding problems, put the complete solution code inside <ANSWER>...</ANSWER>. Do NOT output bare code blocks without the <ANSWER> tag.

    Actions (Choose exactly one):
    1) Answer: <ANSWER>final answer (or option letter like B for multiple choice)</ANSWER>
    """
).strip()


def get_execution_prompt(force_internal: bool = False, mode: str = "normal",
                         has_skills: bool = True, no_code: bool = False) -> str:
    if force_internal:
        if no_code:
            if has_skills:
                return EXECUTION_PROMPT_INTERNAL_NO_CODE
            return EXECUTION_PROMPT_ANSWER_ONLY
        if not has_skills:
            return EXECUTION_PROMPT_CODE_ONLY
        return EXECUTION_PROMPT_INTERNAL
    if mode == "explore_search":
        return EXECUTION_PROMPT_EXPLORE_SEARCH
    if mode == "explore_search_no_skill":
        return EXECUTION_PROMPT_EXPLORE_SEARCH_NO_SKILL
    if no_code:
        return EXECUTION_PROMPT_NO_CODE
    return EXECUTION_PROMPT


FINAL_ANSWER_PROMPT = dedent(
    """\
    Based on the current context, please output your final answer in the following tag:
    <ANSWER>your final answer</ANSWER>
    """
).strip()


# ==================== Parsing Functions ====================

def parse_execution_step(response: str) -> Dict:
    """Parse one round of the execution flow."""
    result = {
        "type": "invalid",
        "thought": "",
        "skill_call": None,
        "run_code": "",
        "answer": "",
        "explore_intent": "",
        "queries": [],
        "page_visit": None,
        "skill_data": None,
        "parse_error": "",
        "raw": response,
    }

    thought_content = _extract_tag_content_from_right(response, "THOUGHT")
    if thought_content is None:
        thought_content = _extract_tag_content_from_right(response, "think")
    if thought_content is not None:
        result["thought"] = thought_content.strip()

    # --- <EXPLORE> (checked before ANSWER to prevent EXPLORE+ANSWER hack) ---
    explore_content = _extract_tag_content_from_right(response, "EXPLORE")
    if explore_content is not None:
        result["type"] = "explore"
        result["explore_intent"] = explore_content.strip()
        return result

    answer_content = _extract_tag_content_from_right(response, "ANSWER")
    if answer_content is not None:
        result["type"] = "answer"
        result["answer"] = answer_content.strip()
        return result

    # --- <CODE> (standalone code execution, takes priority over skill call) ---
    run_code_content = _extract_tag_content_from_right(response, "CODE")
    if run_code_content is not None:
        code = run_code_content.strip()
        if code:
            result["type"] = "run_code"
            result["run_code"] = code
            return result
        result["parse_error"] = "Empty <CODE> tag. Put your Python code inside <CODE>...</CODE>."
        return result
    if _contains_open_tag(response, "CODE") and not _contains_close_tag(response, "CODE"):
        result["parse_error"] = "Detected <CODE> without a closing </CODE> tag."
        return result

    # --- <ACTIVATE_SKILL> (just the skill name, no JSON) ---
    activate_content = _extract_tag_content_from_right(response, "ACTIVATE_SKILL")
    if activate_content is not None:
        skill_name = activate_content.strip()
        if skill_name:
            result["type"] = "skill_call"
            result["skill_call"] = {"skill_name": skill_name}
            return result
        result["parse_error"] = "Empty <ACTIVATE_SKILL> tag. Put the skill_name inside, e.g. <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>."
        return result

    if _contains_open_tag(response, "NEW_GOAL") or _contains_close_tag(response, "NEW_GOAL"):
        result["parse_error"] = "The <NEW_GOAL> step is not supported in this agent flow. Enter exploration and use <QUERY> directly."
        return result

    query_content = _extract_tag_content_from_right(response, "QUERY")
    if query_content is not None:
        result["type"] = "query"
        raw_content = query_content.strip()
        result["queries"] = _parse_query_content(raw_content)[:3]
        return result
    if _contains_open_tag(response, "QUERY") and not _contains_close_tag(response, "QUERY"):
        result["parse_error"] = "Detected <QUERY> without a closing </QUERY> tag."
        return result

    page_visit_content = _extract_tag_content_from_right(response, "PAGE_VISIT")
    if page_visit_content is not None:
        parsed = _parse_page_visit(page_visit_content.strip())
        if parsed:
            result["type"] = "page_visit"
            parsed["urls"] = parsed["urls"][:4]
            result["page_visit"] = parsed
            return result
        result["parse_error"] = (
            'Failed to parse <PAGE_VISIT> JSON. Expected format: '
            '{"urls": ["url1", "url2"], "goal": "what to extract"}'
        )
        return result
    if _contains_open_tag(response, "PAGE_VISIT") and not _contains_close_tag(response, "PAGE_VISIT"):
        result["parse_error"] = "Detected <PAGE_VISIT> without a closing </PAGE_VISIT> tag."
        return result

    skill_content = _extract_tag_content_from_right(response, "GENERATE_SKILL")
    if skill_content is not None:
        parsed = _parse_generate_skill(skill_content.strip())
        if parsed:
            result["type"] = "generate_skill"
            result["skill_data"] = parsed
            return result
        result["parse_error"] = (
            "Failed to parse <GENERATE_SKILL> JSON. Please return valid JSON with at least "
            '"skill_name" and one of "workflow" or "code".'
        )
        return result
    if _contains_open_tag(response, "GENERATE_SKILL") and not _contains_close_tag(response, "GENERATE_SKILL"):
        result["parse_error"] = "Detected <GENERATE_SKILL> without a closing </GENERATE_SKILL> tag."
        return result

    result["parse_error"] = (
        "No valid action tag detected. Return exactly one complete action tag with matching opening and closing tags."
    )
    return result


def _extract_tag_content_from_right(text: str, tag: str) -> Optional[str]:
    """Extract the content of the rightmost complete tag pair.

    This is more robust than a left-to-right non-greedy regex when the model
    casually mentions tags earlier in its reasoning, e.g. inside <think>.
    """
    open_pattern = re.compile(rf"<{re.escape(tag)}>", re.IGNORECASE)
    close_pattern = re.compile(rf"</{re.escape(tag)}>", re.IGNORECASE)
    close_matches = list(close_pattern.finditer(text))
    if not close_matches:
        return None

    for close_match in reversed(close_matches):
        open_matches = list(open_pattern.finditer(text, 0, close_match.start()))
        if not open_matches:
            continue
        open_match = open_matches[-1]
        return text[open_match.end():close_match.start()]
    return None


def _contains_open_tag(text: str, tag: str) -> bool:
    return bool(re.search(rf"<{re.escape(tag)}>", text, re.IGNORECASE))


def _contains_close_tag(text: str, tag: str) -> bool:
    return bool(re.search(rf"</{re.escape(tag)}>", text, re.IGNORECASE))


def _strip_outer_quotes(s: str) -> str:
    """Strip one layer of leading/trailing double quotes from a string."""
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s


def _parse_query_content(raw: str) -> List[str]:
    """Parse QUERY tag content into a list of query strings.

    Primary format: JSON array ["q1", "q2"]
    Fallback: {"querys": ["q1", "q2"]} or plain text -> single-element list
    """
    if not raw:
        return []

    # Try JSON parse — prefer array format
    try:
        obj = json.loads(repair_json(raw))
        if isinstance(obj, list):
            return [_strip_outer_quotes(str(q)) for q in obj if str(q).strip()]
        if isinstance(obj, dict):
            querys = obj.get("querys") or obj.get("queries") or obj.get("query")
            if isinstance(querys, list):
                return [_strip_outer_quotes(str(q)) for q in querys if str(q).strip()]
            if isinstance(querys, str) and querys.strip():
                return [_strip_outer_quotes(querys)]
        if isinstance(obj, str) and obj.strip():
            return [_strip_outer_quotes(obj)]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON array in content
    bracket_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if bracket_match:
        try:
            obj = json.loads(repair_json(bracket_match.group(0)))
            if isinstance(obj, list):
                return [_strip_outer_quotes(str(q)) for q in obj if str(q).strip()]
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: treat as plain text query
    return [_strip_outer_quotes(raw)] if raw.strip() else []


def _parse_page_visit(raw: str) -> Optional[Dict]:
    """Parse PAGE_VISIT tag content.

    Expected format: {"urls": ["url1", "url2"], "goal": "what to extract"}
    """
    if not raw:
        return None

    try:
        obj = json.loads(repair_json(raw))
        if isinstance(obj, dict):
            urls = obj.get("urls") or obj.get("url")
            if isinstance(urls, str):
                urls = [urls]
            if isinstance(urls, list) and urls:
                return {
                    "urls": [str(u).strip() for u in urls if str(u).strip()],
                    "goal": str(obj.get("goal", "")).strip(),
                }
    except (json.JSONDecodeError, ValueError):
        pass

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(repair_json(json_match.group(0)))
            if isinstance(obj, dict):
                urls = obj.get("urls") or obj.get("url")
                if isinstance(urls, str):
                    urls = [urls]
                if isinstance(urls, list) and urls:
                    return {
                        "urls": [str(u).strip() for u in urls if str(u).strip()],
                        "goal": str(obj.get("goal", "")).strip(),
                    }
        except (json.JSONDecodeError, ValueError):
            pass

    return None

def _parse_generate_skill(raw: str) -> Optional[Dict]:
    """Parse GENERATE_SKILL JSON content."""
    if not raw:
        return None

    try:
        obj = json.loads(repair_json(raw))
        if isinstance(obj, dict) and obj.get("skill_name"):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(repair_json(json_match.group(0)))
            if isinstance(obj, dict) and obj.get("skill_name"):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def extract_answer_from_tag(text: str) -> str:
    """Extract content from <ANSWER>...</ANSWER> tag."""
    if not text:
        return ""
    answer = _extract_tag_content_from_right(text, "ANSWER")
    return answer.strip() if answer is not None else ""


def strip_answer_tags(text: str) -> str:
    """Remove <ANSWER> tags from text."""
    return re.sub(r"</?ANSWER>", "", text, flags=re.IGNORECASE)
