# Copyright 2026

import asyncio
import copy
import os
import sys
import tempfile
import time
from typing import Any, Optional
from uuid import uuid4

import yaml

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register


def _candidate_exskill_roots() -> list[str]:
    roots: list[str] = []
    env_path = os.getenv("PYTHONPATH", "")
    for entry in env_path.split(os.pathsep):
        entry = entry.strip()
        if entry:
            roots.append(entry)

    exskill_config_path = os.getenv("EXSKILL_CONFIG_PATH")
    if exskill_config_path:
        roots.append(os.path.dirname(os.path.dirname(os.path.abspath(exskill_config_path))))

    roots.extend(
        [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
            os.path.abspath(os.path.join(os.getcwd(), "..")),
        ]
    )
    return roots


def _ensure_exskill_import_path() -> None:
    for root in _candidate_exskill_roots():
        if not root or not os.path.isdir(root):
            continue
        if not os.path.exists(os.path.join(root, "core", "decision.py")):
            continue
        if root not in sys.path:
            sys.path.insert(0, root)
        return


_ensure_exskill_import_path()

from core.decision import (
    extract_answer_from_tag,
    get_execution_prompt,
    parse_execution_step,
)
from core.explorer import Explorer
from core.skill_manager import SkillManager
from core.skill_schema import (
    normalize_skill,
    skill_code,
    skill_workflow,
)
from llm.client import LLMClient
from tools.code_executor import CodeExecutor
from tools.web_search import WebSearchTool
from tools.web_visit import WebVisitTool, clear_page_cache
from verl.utils.profiler import simple_timer


def _resolve_env_vars(obj: Any) -> Any:
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


def _load_yaml_config(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return _resolve_env_vars(config)


@register("exskill_agent")
class ExSkillAgentLoop(AgentLoopBase):
    """Run the original ExSkill state machine on top of verl's async rollout server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rollout_cfg = self.config.actor_rollout_ref.rollout
        multi_turn_cfg = rollout_cfg.multi_turn
        self.response_length = rollout_cfg.response_length
        self.max_assistant_turns = multi_turn_cfg.max_assistant_turns or 12
        self.max_tool_response_length = multi_turn_cfg.max_tool_response_length
        self.force_internal = bool(getattr(multi_turn_cfg, "force_internal", False))

        exskill_config_path = getattr(multi_turn_cfg, "exskill_config_path", None) or os.getenv("EXSKILL_CONFIG_PATH")
        self.exskill_config_path = exskill_config_path
        self.exskill_config = _load_yaml_config(exskill_config_path)

        agent_cfg = self.exskill_config.get("agent", {})
        self.explore_max_searches = int(agent_cfg.get("explore_max_searches", 3) or 3)
        self.explore_max_total_searches = int(agent_cfg.get("explore_max_total_searches", 4) or 4)
        self.explore_max_page_visits = int(agent_cfg.get("explore_max_page_visits", 4) or 4)
        self.fetch_top_k = int(agent_cfg.get("fetch_top_k", 3) or 3)

        self.web_search = None
        self.web_visit = None
        self.code_executor = None
        if self.exskill_config:
            compress_profile = self.exskill_config.get("web_visit", {}).get("compress_llm", "qwen-turbo")
            compress_llm = LLMClient.from_config(self.exskill_config, profile_name=compress_profile)
            self.web_search = WebSearchTool.from_config(self.exskill_config)
            self.web_visit = WebVisitTool.from_config(self.exskill_config, compress_llm=compress_llm)
            self.code_executor = CodeExecutor.from_config(self.exskill_config)

            # Set up API monitoring callback on LLM clients
            def _api_callback(profile_name: str, model: str, success: bool) -> None:
                monitor = getattr(self, "_current_agent_loop_api_monitor", None)
                if monitor is not None:
                    monitor.append({"profile": profile_name, "model": model, "success": success})

            compress_llm.set_api_callback(_api_callback)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        agent_loop_api_monitor: list[dict[str, Any]] = []
        self._current_agent_loop_api_monitor = agent_loop_api_monitor
        messages = list(kwargs["raw_prompt"])
        extra_info = dict(kwargs.get("extra_info", {}) or {})
        skillset_meta = dict(extra_info.get("skillset_meta") or {})
        explore_required = bool(skillset_meta.get("explore_required", extra_info.get("explore_required", False)))
        available_skills = [normalize_skill(dict(skill)) for skill in extra_info.get("available_skills", [])]
        skill_manager = self._make_skill_manager(available_skills)
        skill_manager.skills = [
            normalize_skill(skill)
            for skill in available_skills
            if isinstance(skill, dict) and not normalize_skill(skill).get("builtin")
        ]
        explorer = self._make_explorer(skill_manager)

        clear_page_cache()

        generated_skills: list[dict[str, Any]] = []
        code_context: list[str] = []
        skill_code_retry_count: int = 0
        actions: list[str] = []
        called_skill_names: list[str] = []
        search_queries: list[str] = []
        explore_goals: list[str] = []
        page_visit_blocks: list[str] = []
        exploration_phase: Optional[str] = None
        raw_explore_intent = ""
        exploration_goal = ""
        exploration_candidates: list[str] = []
        explore_count = 0
        total_search_count = 0
        explore_turns = 0
        page_visit_count = 0
        total_page_visit_count = 0
        page_visit_success_count = 0
        page_visit_failed_count = 0
        page_visit_forbidden_count = 0
        explore_token_length = 0
        used_explore = False
        final_answer = ""

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)

        request_id = uuid4().hex
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        assistant_turns = 0
        user_turns = 0
        metrics: dict[str, Any] = {
            "generate_sequences": 0.0,
            "tool_calls": 0.0,
            "num_preempted": 0,
            "search": 0.0,
            "page_visit": 0.0,
            "page_visit_fetch": 0.0,
            "page_visit_compress": 0.0,
            "code_execution": 0.0,
            "tokenization": 0.0,
        }
        routed_experts = None
        # Per-trajectory timing for end-of-trajectory print
        _traj_timing: dict[str, float] = {
            "generate_sequences": 0.0,
            "decode": 0.0,
            "parse_step": 0.0,
            "search": 0.0,
            "page_visit": 0.0,
            "page_visit_fetch": 0.0,
            "page_visit_compress": 0.0,
            "code_execution": 0.0,
            "tokenization": 0.0,
            "skill_registration": 0.0,
        }
        _traj_start = time.time()

        while len(response_mask) < self.response_length and assistant_turns < self.max_assistant_turns:
            with simple_timer("generate_sequences", metrics):
                _t0 = time.time()
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    video_data=videos,
                )
                _traj_timing["generate_sequences"] += time.time() - _t0

            metrics["num_preempted"] = metrics.get("num_preempted", 0) + (
                output.num_preempted if output.num_preempted is not None else 0
            )

            assistant_turns += 1
            routed_experts = output.routed_experts
            new_response_ids = output.token_ids
            prompt_ids += new_response_ids
            response_mask += [1] * len(new_response_ids)
            if output.log_probs:
                response_logprobs += output.log_probs

            _t0 = time.time()
            assistant_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(new_response_ids, skip_special_tokens=True)
            )
            _traj_timing["decode"] += time.time() - _t0
            messages.append({"role": "assistant", "content": assistant_text})

            _t0 = time.time()
            step = parse_execution_step(assistant_text)
            _traj_timing["parse_step"] += time.time() - _t0
            actions.append(step["type"])

            # Track explore-phase metrics
            if step["type"] in ("explore", "query", "page_visit", "generate_skill"):
                explore_token_length += len(new_response_ids)
                if step["type"] in ("query", "page_visit", "generate_skill"):
                    explore_turns += 1

            if step["type"] == "answer":
                if exploration_phase is not None:
                    feedback = (
                        "You are currently in exploration mode. You must first generate a skill before you can answer.\n"
                        "Use <QUERY>...</QUERY> for more search, "
                        "<PAGE_VISIT>{\"urls\": [...], \"goal\": \"...\"}</PAGE_VISIT> to visit pages, "
                        "or <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                    )
                else:
                    final_answer = step.get("answer", "")
                    break

            elif step["type"] == "invalid":
                feedback = self._invalid_feedback(step, exploration_phase)

            elif step["type"] == "explore":
                used_explore = True
                if self.force_internal:
                    feedback = (
                        "Exploration is disabled. Please answer using your existing knowledge and available skills.\n"
                        "Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, <CODE>...</CODE>, or <ANSWER>...</ANSWER>."
                    )
                elif exploration_phase is not None:
                    feedback = (
                        "You have already entered exploration mode.\n"
                        "Use <QUERY>...</QUERY> to continue searching, "
                        "<PAGE_VISIT>{\"urls\": [...], \"goal\": \"...\"}</PAGE_VISIT> to visit pages, "
                        "or <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                    )
                else:
                    raw_explore_intent = step.get("explore_intent", "").strip()
                    exploration_goal = raw_explore_intent or str(extra_info.get("question", "")).strip()
                    if exploration_goal:
                        explore_goals.append(exploration_goal)
                    exploration_phase = "search"
                    explore_count = 0
                    exploration_candidates = self._candidate_skill_names(called_skill_names, available_skills)
                    feedback = get_execution_prompt(
                        force_internal=self.force_internal,
                        mode="explore_search",
                    ).format(
                        exploration_goal=exploration_goal or "(empty)",
                        candidate_skills=", ".join(exploration_candidates) if exploration_candidates else "(none)",
                    )

            elif step["type"] == "query":
                if exploration_phase is None:
                    feedback = "You are not in exploration mode yet. Use <EXPLORE>...</EXPLORE> first if you need exploration."
                elif total_search_count >= self.explore_max_total_searches:
                    feedback = (
                        f"Total search limit reached ({self.explore_max_total_searches} searches across all explorations).\n"
                        "You must now generate a reusable skill from the information gathered so far.\n"
                        "Use <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                    )
                elif explore_count >= self.explore_max_searches:
                    feedback = (
                        f"Search limit reached ({self.explore_max_searches} searches).\n"
                        "You must now generate a reusable skill from the information gathered so far.\n"
                        "Use <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                    )
                elif explorer is None:
                    feedback = self._missing_runtime_feedback("search")
                else:
                    queries = step.get("queries", [])
                    if not queries:
                        feedback = 'No queries provided. Use <QUERY>["search keyword"]</QUERY>.'
                    else:
                        search_queries.extend(queries)
                        with simple_timer("search", metrics):
                            _t0 = time.time()
                            search_data = await explorer.execute_search(queries, knowledge_gaps=exploration_goal or "; ".join(queries))
                            _traj_timing["search"] += time.time() - _t0
                        if search_data.get("search_results"):
                            explore_count += 1
                            total_search_count += 1
                        formatted_info = search_data.get("formatted_info", "No results.")
                        queries_str = "; ".join(queries)
                        feedback = (
                            f"Exploration goal: {exploration_goal}\n"
                            f"Search results for queries: {queries_str}\n\n"
                            f"{formatted_info}\n\n"
                            "You may search more with <QUERY>...</QUERY>, "
                            "visit specific pages with <PAGE_VISIT>{\"urls\": [...], \"goal\": \"...\"}</PAGE_VISIT>, "
                            "or generate/evolve a reusable skill with <GENERATE_SKILL>{...}</GENERATE_SKILL>.\n"
                            "If you are extending an existing skill, reuse its exact skill_name."
                        )

            elif step["type"] == "page_visit":
                if exploration_phase is None:
                    feedback = "You are not in exploration mode yet. Use <EXPLORE>...</EXPLORE> first if you need exploration."
                elif total_page_visit_count >= self.explore_max_page_visits:
                    feedback = (
                        f"Page visit limit reached ({self.explore_max_page_visits} visits).\n"
                        "Use the information you already have to generate a skill.\n"
                        "Use <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                    )
                elif explorer is None:
                    feedback = self._missing_runtime_feedback("page_visit")
                else:
                    page_visit_data = step.get("page_visit") or {}
                    urls = page_visit_data.get("urls", [])
                    goal = page_visit_data.get("goal", exploration_goal or "")
                    if not urls:
                        feedback = 'No URLs provided. Use <PAGE_VISIT>{"urls": ["url1", "url2"], "goal": "what to extract"}</PAGE_VISIT>.'
                    else:
                        page_visit_count += len(urls)
                        total_page_visit_count += 1
                        with simple_timer("page_visit", metrics):
                            _t0 = time.time()
                            visit_data = await explorer.execute_page_visit(urls, goal=goal)
                            _pv_wall = time.time() - _t0
                        _traj_timing["page_visit"] += _pv_wall
                        if self.web_visit is not None:
                            _traj_timing["page_visit_fetch"] += self.web_visit._fetch_time
                            _traj_timing["page_visit_compress"] += self.web_visit._compress_time
                            metrics["page_visit_fetch"] = metrics.get("page_visit_fetch", 0.0) + self.web_visit._fetch_time
                            metrics["page_visit_compress"] = metrics.get("page_visit_compress", 0.0) + self.web_visit._compress_time
                        # Track page visit success/failure/forbidden
                        for vr in visit_data.get("visited_pages", []):
                            src = vr.get("fetch_source")
                            if src == "failed":
                                page_visit_failed_count += 1
                            elif src == "forbidden":
                                page_visit_forbidden_count += 1
                            else:
                                page_visit_success_count += 1
                        formatted_info = visit_data.get("formatted_info", "No content extracted.")
                        feedback = (
                            f"Page visit results for {len(urls)} URL(s):\n\n"
                            f"{formatted_info}\n\n"
                            "You may search more with <QUERY>...</QUERY>, "
                            "visit more pages with <PAGE_VISIT>{\"urls\": [...], \"goal\": \"...\"}</PAGE_VISIT>, "
                            "or generate/evolve a reusable skill with <GENERATE_SKILL>{...}</GENERATE_SKILL>.\n"
                            "If you are extending an existing skill, reuse its exact skill_name."
                        )
                        page_visit_blocks.append(feedback)

            elif step["type"] == "generate_skill":
                if exploration_phase is None:
                    feedback = (
                        "You haven't explored yet. "
                        + (
                            "Exploration is disabled in this run. Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, <CODE>...</CODE>, or <ANSWER>...</ANSWER>."
                            if self.force_internal
                            else "Use <EXPLORE>...</EXPLORE> to enter exploration first, then search and generate a skill."
                        )
                    )
                elif explore_count == 0:
                    feedback = (
                        "You must perform at least one search before generating a skill. "
                        "Use <QUERY>[\"search keyword\"]</QUERY> to search for information first."
                    )
                elif explorer is None:
                    feedback = self._missing_runtime_feedback("skill_generation")
                else:
                    _t0 = time.time()
                    new_skill = await self.loop.run_in_executor(
                        None,
                        lambda: explorer.parse_and_register_skill(
                            step.get("skill_data") or {},
                            current_skills=available_skills,
                            preferred_skill_names=exploration_candidates,
                        ),
                    )
                    if not new_skill:
                        feedback = (
                            "Failed to parse the skill. Please provide valid JSON with at least "
                            '"skill_name" and one of "workflow" or "code".\n'
                            "Use <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                        )
                    else:
                        normalized_skill = normalize_skill(new_skill)
                        code = skill_code(normalized_skill)

                        # Validate skill code syntax before injection
                        skill_code_syntax_error = None
                        if code and code.strip():
                            try:
                                compile(code.strip(), "<skill>", "exec")
                            except SyntaxError as e:
                                skill_code_syntax_error = e

                        if skill_code_syntax_error and skill_code_retry_count < 2:
                            # Give model a chance to regenerate with fixed code
                            skill_code_retry_count += 1
                            print(f"[skill_code] SyntaxError (attempt {skill_code_retry_count}/2): "
                                  f"{skill_code_syntax_error}", flush=True)
                            feedback = (
                                f"Skill code has a syntax error: {skill_code_syntax_error}\n\n"
                                "Please regenerate the complete skill with fixed code. "
                                "Make sure all parentheses, brackets, and quotes are properly closed.\n"
                                "Use <GENERATE_SKILL>{...}</GENERATE_SKILL> with the corrected code."
                            )
                        else:
                            # Either code is valid, no code, or max retries exhausted
                            self._merge_skill_into_available(available_skills, normalized_skill)
                            generated_skills.append(normalized_skill)
                            skill_manager.skills = [
                                normalize_skill(skill)
                                for skill in available_skills
                                if isinstance(skill, dict) and not normalize_skill(skill).get("builtin")
                            ]

                            skill_code_valid = True
                            if skill_code_syntax_error:
                                # Max retries exhausted, register workflow only
                                skill_code_valid = False
                                print(f"[skill_code] SyntaxError persists after {skill_code_retry_count} "
                                      f"retries, registering workflow only", flush=True)
                            elif code and code.strip() not in code_context:
                                code_context.append(code.strip())

                            _exhausted_retries = skill_code_retry_count
                            skill_code_retry_count = 0

                            blocks = [f"### {normalized_skill.get('skill_name', 'Unknown')}"]
                            workflow = skill_workflow(normalized_skill)
                            if workflow:
                                blocks.append("Workflow:\n" + workflow)
                            if code and skill_code_valid:
                                blocks.append(
                                    "Code (loaded and available via <CODE>):\n\n"
                                    f"```python\n{code}\n```"
                                )
                            elif code and not skill_code_valid:
                                blocks.append(
                                    "Note: The generated code still had syntax errors after "
                                    f"{_exhausted_retries} attempts and was NOT loaded. "
                                    "You can still use the workflow above. "
                                    "If you need to compute, write your own code in <CODE>...</CODE>."
                                )

                            exploration_phase = None
                            raw_explore_intent = ""
                            exploration_goal = ""
                            exploration_candidates = []
                            explore_count = 0

                            feedback = (
                                (
                                    "Skill evolved and registered (auto-activated):\n\n"
                                    if normalized_skill.get("action") == "evolve"
                                    else "Skill generated and registered (auto-activated):\n\n"
                                )
                                + "\n\n".join(blocks)
                                + "\n\nThis skill is already activated — workflow is loaded and code is available for <CODE>. "
                                "No need to call <ACTIVATE_SKILL> for it again.\n"
                                + self._next_step_actions_hint(code_context)
                            )
                    _traj_timing["skill_registration"] += time.time() - _t0

            elif step["type"] == "run_code":
                if exploration_phase is not None:
                    feedback = (
                        "You are currently in exploration mode. You must first generate a skill before running code.\n"
                        "Use <QUERY>...</QUERY> for more search, "
                        "<PAGE_VISIT>{\"urls\": [...], \"goal\": \"...\"}</PAGE_VISIT> to visit pages, "
                        "or <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                    )
                else:
                    code = step.get("run_code", "").strip()
                    if not code:
                        feedback = "No code provided. Use <CODE>your Python code</CODE>."
                    elif self.code_executor is None:
                        feedback = self._missing_runtime_feedback("code_execution")
                    else:
                        full_code = "\n\n".join(code_context + [f"# --- User code ---\n{code}"]) if code_context else code
                        with simple_timer("code_execution", metrics):
                            _t0 = time.time()
                            result = await self.loop.run_in_executor(
                                None, lambda: self.code_executor.execute(full_code)
                            )
                            _traj_timing["code_execution"] += time.time() - _t0
                        if result["success"]:
                            stdout = result["stdout"].strip()
                            if stdout:
                                feedback = (
                                    "Code execution feedback:\n"
                                    f"Code executed successfully.\nOutput:\n{stdout}\n\n"
                                    "Continue with your next step.\n"
                                    + self._next_step_actions_hint(code_context)
                                )
                            else:
                                feedback = (
                                    "Code execution feedback:\n"
                                    "Code executed successfully, but produced no output.\n"
                                    "Make sure to use print() to output the results you want to see. "
                                    "For example, instead of just computing a value, write print(result) to display it.\n\n"
                                    "Continue with your next step.\n"
                                    + self._next_step_actions_hint(code_context)
                                )
                        else:
                            # Layer 2: if skill code caused the failure, remove it and retry
                            if code_context:
                                retry_result = await self.loop.run_in_executor(
                                    None, lambda c=code: self.code_executor.execute(c)
                                )
                                if retry_result["success"]:
                                    print(f"[skill_code] Skill code caused execution failure, "
                                          f"removed {len(code_context)} skill code block(s) from context", flush=True)
                                    code_context.clear()
                                    result = retry_result
                                    stdout = result["stdout"].strip()
                                    if stdout:
                                        feedback = (
                                            "Code execution feedback:\n"
                                            "Note: Previously loaded skill code had errors and was removed.\n"
                                            f"Code executed successfully.\nOutput:\n{stdout}\n\n"
                                            "Continue with your next step. "
                                            "Define any needed functions directly in your <CODE> block.\n"
                                            + self._next_step_actions_hint(code_context)
                                        )
                                    else:
                                        feedback = (
                                            "Code execution feedback:\n"
                                            "Note: Previously loaded skill code had errors and was removed.\n"
                                            "Code executed successfully, but produced no output.\n"
                                            "Make sure to use print() to output the results.\n\n"
                                            "Continue with your next step.\n"
                                            + self._next_step_actions_hint(code_context)
                                        )
                                else:
                                    feedback = (
                                        "Code execution feedback:\n"
                                        f"Code execution failed.\nstdout: {result['stdout']}\nstderr: {result['stderr']}\n\n"
                                        "Continue with your next step.\n"
                                        + self._next_step_actions_hint(code_context)
                                    )
                            else:
                                feedback = (
                                    "Code execution feedback:\n"
                                    f"Code execution failed.\nstdout: {result['stdout']}\nstderr: {result['stderr']}\n\n"
                                    "Continue with your next step.\n"
                                    + self._next_step_actions_hint(code_context)
                                )

            elif step["type"] == "skill_call":
                if exploration_phase is not None:
                    feedback = (
                        "You are currently in exploration mode. You must first generate a skill before activating skills.\n"
                        "Use <QUERY>...</QUERY> for more search, "
                        "<PAGE_VISIT>{\"urls\": [...], \"goal\": \"...\"}</PAGE_VISIT> to visit pages, "
                        "or <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                    )
                else:
                    feedback, used_skill_name = await self._execute_activate_skill(
                        step.get("skill_call") or {},
                        available_skills,
                        code_context,
                        skill_manager,
                    )
                    if used_skill_name:
                        called_skill_names.append(used_skill_name)

            else:
                feedback = (
                    "Format error.\n"
                    f"{step.get('parse_error', 'No valid action tag detected.')}\n"
                    "Return exactly one complete action tag."
                )

            await self._append_user_feedback(messages, feedback)
            with simple_timer("tokenization", metrics):
                _t0 = time.time()
                feedback_ids = await self.apply_chat_template([messages[-1]], remove_system_prompt=True)
                _traj_timing["tokenization"] += time.time() - _t0
            prompt_ids += feedback_ids
            response_mask += [0] * len(feedback_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(feedback_ids)
            user_turns += 1

            if len(response_mask) >= self.response_length:
                break

        if not final_answer:
            final_answer = extract_answer_from_tag(" ".join(msg["content"] for msg in messages if msg["role"] == "assistant"))

        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_token_ids = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids

        metrics["assistant_turns"] = assistant_turns
        metrics["user_turns"] = user_turns

        # Print per-trajectory timing breakdown
        _traj_total = time.time() - _traj_start
        _tracked = sum(v for k, v in _traj_timing.items() if k not in ("page_visit",))
        _other = max(0.0, _traj_total - _tracked)
        _traj_timing["other"] = _other
        print(f"\n[Trajectory Timing] total={_traj_total:.1f}s  actions={actions}", flush=True)
        for _k in ("generate_sequences", "decode", "parse_step", "search", "page_visit", "page_visit_fetch", "page_visit_compress",
                    "code_execution", "skill_registration", "tokenization", "other"):
            _v = _traj_timing.get(_k, 0.0)
            _pct = _v / _traj_total * 100 if _traj_total > 0 else 0
            print(f"  {_k:30s} = {_v:6.1f}s ({_pct:5.1f}%)", flush=True)
        print(flush=True)

        # --- Skill quality evaluation ---
        skill_quality_eval_results = None
        skill_quality_eval_meta = None

        async def _do_skill_quality():
            if not generated_skills:
                return {"score": None, "skipped": True, "reason": "no_skill_generated", "per_skill": []}
            sq_enabled = os.getenv("SKILL_QUALITY_ENABLED", "true").lower() in ("true", "1", "yes")
            if not sq_enabled:
                return {"score": None, "skipped": True, "reason": "disabled", "per_skill": []}
            try:
                from evaluation.skill_quality_judge import evaluate_skill_quality
            except Exception as e:
                print(f"[skill_quality_eval] import failed: {e}", flush=True)
                return {"score": None, "skipped": True, "reason": f"import_failed:{e}", "per_skill": []}
            try:
                return await evaluate_skill_quality(
                    question=str(extra_info.get("question") or "").strip(),
                    ground_truth=str(extra_info.get("answer") or "").strip(),
                    student_knowledge=str(extra_info.get("student_knowledge") or ""),
                    page_visits=list(page_visit_blocks),
                    explore_goals=list(explore_goals),
                    generated_skills=generated_skills,
                )
            except Exception as e:
                print(f"[skill_quality_eval] ERROR: {e}", flush=True)
                return {"score": None, "skipped": True, "reason": f"error:{e}", "per_skill": []}

        _eval_t0 = time.time()
        skill_quality_eval_results = await _do_skill_quality()
        _eval_elapsed = round(time.time() - _eval_t0, 2)

        if skill_quality_eval_results is not None:
            skill_quality_eval_meta = {
                "elapsed_sec": skill_quality_eval_results.get("elapsed_sec", 0.0),
                "n_skills": len((skill_quality_eval_results or {}).get("per_skill", [])),
                "skipped": bool((skill_quality_eval_results or {}).get("skipped", False)),
                "reason": (skill_quality_eval_results or {}).get("reason", ""),
            }
            print(
                f"[skill_quality_eval] elapsed={skill_quality_eval_meta['elapsed_sec']}s "
                f"n_skills={skill_quality_eval_meta['n_skills']} "
                f"skipped={skill_quality_eval_meta['skipped']} "
                f"score={(skill_quality_eval_results or {}).get('score')}",
                flush=True,
            )

        # Aggregate API monitor stats by profile (after skill_reuse eval so frozen LLM calls are included)
        api_by_profile: dict[str, dict[str, int]] = {}
        for stat in agent_loop_api_monitor:
            key = stat.get("profile") or stat.get("model") or "unknown"
            if key not in api_by_profile:
                api_by_profile[key] = {"total": 0, "success": 0}
            api_by_profile[key]["total"] += 1
            if stat.get("success"):
                api_by_profile[key]["success"] += 1
        self._current_agent_loop_api_monitor = None

        return AgentLoopOutput(
            prompt_ids=prompt_token_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                routed_experts[: len(prompt_token_ids) + self.response_length]
                if routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            reward_score=None,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields={
                "conversation_messages": messages,
                "actions": actions,
                "generated_skills": generated_skills,
                "search_queries": search_queries,
                "explore_goals": explore_goals,
                "page_visit_blocks": page_visit_blocks,
                "final_answer": final_answer,
                "explore_turns": explore_turns,
                "search_count": total_search_count,
                "page_visit_count": total_page_visit_count,
                "page_visit_success_count": page_visit_success_count,
                "page_visit_failed_count": page_visit_failed_count,
                "used_explore": used_explore,
                "explore_required": explore_required,
                "page_visit_forbidden_count": page_visit_forbidden_count,
                "explore_token_length": explore_token_length,
                "agent_loop_api_calls_total": len(agent_loop_api_monitor),
                "agent_loop_api_calls_success": sum(1 for s in agent_loop_api_monitor if s.get("success")),
                "agent_loop_api_calls_failed": len(agent_loop_api_monitor)
                - sum(1 for s in agent_loop_api_monitor if s.get("success")),
                "agent_loop_api_by_profile": api_by_profile,
                "skill_quality_eval_results": skill_quality_eval_results,
                "skill_quality_eval_meta": skill_quality_eval_meta,
            },
        )

    async def _append_user_feedback(self, messages: list[dict[str, Any]], feedback: str) -> None:
        if len(feedback) > self.max_tool_response_length:
            feedback = feedback[: self.max_tool_response_length] + "\n...[truncated]"
        messages.append({"role": "user", "content": feedback})

    def _make_skill_manager(self, available_skills: list[dict[str, Any]]) -> SkillManager:
        config = copy.deepcopy(self.exskill_config or {})
        config.setdefault("skill", {})
        temp_lib = os.path.join(tempfile.gettempdir(), f"verl_exskill_{uuid4().hex}.json")
        config["skill"]["library_path"] = temp_lib
        config["skill"].setdefault("max_skills", 200)
        config["skill"].setdefault("similarity_threshold", 0.8)
        skill_manager = SkillManager.from_config(config)
        skill_manager.skills = [
            normalize_skill(skill)
            for skill in available_skills
            if isinstance(skill, dict) and not normalize_skill(skill).get("builtin")
        ]
        return skill_manager

    def _make_explorer(self, skill_manager: SkillManager) -> Optional[Explorer]:
        if not (self.web_search and self.web_visit):
            return None
        return Explorer(
            skill_manager,
            self.web_search,
            self.web_visit,
            fetch_top_k=self.fetch_top_k,
        )

    def _invalid_feedback(self, step: dict[str, Any], exploration_phase: Optional[str]) -> str:
        parse_error = step.get("parse_error") or "Invalid action format."
        if exploration_phase is not None:
            if "closing </QUERY>" in parse_error or "closing </GENERATE_SKILL>" in parse_error or "closing </PAGE_VISIT>" in parse_error:
                phase_error = parse_error
            elif "Failed to parse <GENERATE_SKILL>" in parse_error or "Failed to parse <PAGE_VISIT>" in parse_error:
                phase_error = parse_error
            else:
                phase_error = "Invalid exploration-step format."
            return (
                "Format error in the exploration-search step.\n"
                f"{phase_error}\n"
                "Use exactly one complete <QUERY>...</QUERY>, "
                "<PAGE_VISIT>{\"urls\": [...], \"goal\": \"...\"}</PAGE_VISIT>, "
                "or <GENERATE_SKILL>{...}</GENERATE_SKILL> tag."
            )
        valid_actions = (
            "Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, <CODE>...</CODE>, or <ANSWER>...</ANSWER>."
            if self.force_internal
            else "Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, <CODE>...</CODE>, <ANSWER>...</ANSWER>, or <EXPLORE>...</EXPLORE>."
        )
        return f"Format error.\n{parse_error}\n{valid_actions}"

    def _candidate_skill_names(self, called_skill_names: list[str], available_skills: list[dict[str, Any]]) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for skill_name in called_skill_names:
            normalized_name = self._normalize_skill_name(skill_name)
            if normalized_name and normalized_name not in seen:
                seen.add(normalized_name)
                candidates.append(normalized_name)
        for skill in available_skills:
            normalized_skill = normalize_skill(skill)
            if normalized_skill.get("builtin"):
                continue
            skill_name = self._normalize_skill_name(normalized_skill.get("skill_name", ""))
            if skill_name and skill_name not in seen:
                seen.add(skill_name)
                candidates.append(skill_name)
        return candidates

    def _merge_skill_into_available(self, available_skills: list[dict[str, Any]], new_skill: dict[str, Any]) -> None:
        updated = False
        if new_skill.get("action") == "evolve" and new_skill.get("skill_id"):
            for idx, existing in enumerate(available_skills):
                if normalize_skill(existing).get("skill_id") == new_skill["skill_id"]:
                    available_skills[idx] = new_skill
                    updated = True
                    break
        if not updated and new_skill.get("action") == "evolve":
            target_name = self._normalize_skill_name(new_skill.get("skill_name", ""))
            for idx, existing in enumerate(available_skills):
                if self._normalize_skill_name(normalize_skill(existing).get("skill_name", "")) == target_name:
                    available_skills[idx] = new_skill
                    updated = True
                    break
        if not updated:
            recalled_ids = {normalize_skill(skill).get("skill_id") for skill in available_skills}
            if new_skill.get("skill_id") and new_skill["skill_id"] not in recalled_ids:
                available_skills.append(new_skill)

    async def _execute_activate_skill(
        self,
        call: dict[str, Any],
        available_skills: list[dict[str, Any]],
        code_context: list[str],
        skill_manager: SkillManager,
    ) -> tuple[str, Optional[str]]:
        skill_name = str(call.get("skill_name", "")).strip()
        matched_skill = self._find_skill_by_name(skill_name, available_skills)
        if not matched_skill:
            available = [
                normalize_skill(skill).get("skill_name", "")
                for skill in available_skills
                if isinstance(skill, dict) and not normalize_skill(skill).get("builtin")
            ]
            return (
                f"Skill '{skill_name}' not found. Available skill_name values: {available}. Please try again.",
                None,
            )

        matched_skill = normalize_skill(matched_skill)
        normalized_name = self._normalize_skill_name(matched_skill.get("skill_name") or matched_skill.get("name"))

        sections: list[str] = []
        workflow = skill_workflow(matched_skill)
        code = skill_code(matched_skill).strip()
        if workflow:
            sections.append(f"Workflow from skill '{matched_skill['skill_name']}':\n\n{workflow}")
        if code:
            try:
                compile(code, "<skill>", "exec")
            except SyntaxError as e:
                print(f"[skill_code] SyntaxError in activated skill code, skipping: {e}", flush=True)
                code = ""
        if code:
            if code not in code_context:
                code_context.append(code)
            sections.append(
                f"Code from skill '{matched_skill['skill_name']}' loaded. The following function is now available:\n\n"
                f"```python\n{code}\n```\n\n"
                "Use <CODE> to call the function directly. No import is needed."
            )

        if matched_skill.get("skill_id"):
            await self.loop.run_in_executor(
                None, lambda sid=matched_skill["skill_id"]: skill_manager.update_stats(sid, success=True)
            )

        if not sections:
            sections.append("This skill has neither workflow nor code content.")
        return (
            "Execution feedback:\n"
            + "\n\n".join(sections)
            + "\n\nContinue with your next step.\n"
            + self._next_step_actions_hint(code_context),
            normalized_name,
        )

    def _find_skill_by_name(self, name: Optional[str], skills: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not skills or not name:
            return None
        target_skill_name = self._normalize_skill_name(name)
        if not target_skill_name:
            return None
        for skill in skills:
            normalized_skill = normalize_skill(skill)
            skill_name = self._normalize_skill_name(normalized_skill.get("skill_name", ""))
            if skill_name and skill_name == target_skill_name:
                return normalized_skill
        return None

    def _normalize_skill_name(self, value: Optional[str]) -> str:
        text = str(value or "").strip().lower()
        parts = []
        last_was_sep = False
        for char in text:
            if char.isalnum():
                parts.append(char)
                last_was_sep = False
            elif not last_was_sep:
                parts.append("_")
                last_was_sep = True
        return "".join(parts).strip("_")

    def _next_step_actions_hint(self, code_context: Optional[list[str]]) -> str:
        base = (
            "Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, <CODE>...</CODE>, or <ANSWER>...</ANSWER>."
            if self.force_internal
            else "Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, <CODE>...</CODE>, <ANSWER>...</ANSWER>, or <EXPLORE>...</EXPLORE>."
        )
        if code_context:
            base += (
                "\nYou can use <CODE>...</CODE> to directly call functions loaded from skill code. No import is needed."
            )
        return base

    def _missing_runtime_feedback(self, capability: str) -> str:
        config_path = self.exskill_config_path or "unset"
        return (
            f"ExSkill runtime is not configured for {capability}. "
            f"Set actor_rollout_ref.rollout.multi_turn.exskill_config_path or EXSKILL_CONFIG_PATH. "
            f"Current value: {config_path}."
        )
