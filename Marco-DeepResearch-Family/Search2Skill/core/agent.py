import json
import re
import threading
import time
from typing import Dict, List, Optional

from llm.client import LLMClient
from core.skill_manager import SkillManager
from core.explorer import Explorer
from core.skill_schema import normalize_skill, skill_code, skill_workflow
from core.decision import (
    get_execution_prompt,
    FINAL_ANSWER_PROMPT,
    parse_execution_step,
    extract_answer_from_tag,
    strip_answer_tags,
)

# Global page visit counters (thread-safe, for eval monitoring)
_page_visit_lock = threading.Lock()
_page_visit_success = 0
_page_visit_failed = 0
_page_visit_forbidden = 0

# When True, suppress per-round print output (for clean eval progress)
_quiet = False


def set_quiet(quiet: bool = True):
    """Enable/disable per-round verbose output."""
    global _quiet
    _quiet = quiet


def _log(msg: str):
    """Print only when not in quiet mode."""
    if not _quiet:
        print(msg)


def get_page_visit_stats() -> dict:
    """Return global page visit success/failure/forbidden counts."""
    with _page_visit_lock:
        return {"success": _page_visit_success, "failed": _page_visit_failed,
                "forbidden": _page_visit_forbidden,
                "total": _page_visit_success + _page_visit_failed + _page_visit_forbidden}


def reset_page_visit_stats():
    """Reset global page visit counters."""
    global _page_visit_success, _page_visit_failed, _page_visit_forbidden
    with _page_visit_lock:
        _page_visit_success = 0
        _page_visit_failed = 0
        _page_visit_forbidden = 0
from tools.web_search import WebSearchTool
from tools.web_visit import WebVisitTool, clear_page_cache
from tools.code_executor import CodeExecutor


class ExplorationAgent:
    """Exploration-driven skill generation agent.

    Every execution round the model can choose:
      1) ACTIVATE_SKILL – activate an existing skill (loads knowledge/code)
      2) CODE       – execute Python code (standalone, not a skill)
      3) ANSWER         – output final answer
      4) EXPLORE        – declare a missing capability and enter exploration mode
      5) QUERY          – run web search queries (supports batch JSON array)
      6) PAGE_VISIT     – visit selected URLs from search results
      7) GENERATE_SKILL – generate or evolve a skill from exploration results

    State machine: normal -> explore_search -> normal
    """

    def __init__(self, llm: LLMClient, skill_manager: SkillManager,
                 web_search: WebSearchTool, web_visit: WebVisitTool,
                 code_executor: Optional[CodeExecutor] = None,
                 max_rounds: int = 5, explore_max_searches: int = 3,
                 force_internal: bool = False,
                 fetch_top_k: int = 3,
                 skip_skill_generation: bool = False,
                 teacher_llm: Optional[LLMClient] = None,
                 abort_on_explore: bool = False,
                 skill_reference_only: bool = False,
                 force_first_explore: bool = False):
        self.llm = llm
        self.skill_manager = skill_manager
        self.code_executor = code_executor
        self.max_rounds = max_rounds
        self.explore_max_searches = explore_max_searches
        self.force_internal = force_internal
        self.skip_skill_generation = skip_skill_generation
        self.teacher_llm = teacher_llm
        self.abort_on_explore = abort_on_explore
        self.skill_reference_only = skill_reference_only
        self.force_first_explore = force_first_explore
        self.explorer = Explorer(
            skill_manager, web_search, web_visit,
            fetch_top_k=fetch_top_k,
        )

    @classmethod
    def from_config(cls, config, force_internal: bool = False) -> "ExplorationAgent":
        agent_cfg = config["agent"]
        wv_cfg = config.get("web_visit", {})
        compress_profile = wv_cfg.get("compress_llm", "qwen-turbo")
        compress_llm = LLMClient.from_config(config, profile_name=compress_profile)
        return cls(
            llm=LLMClient.from_config(config),
            skill_manager=SkillManager.from_config(config),
            web_search=WebSearchTool.from_config(config),
            web_visit=WebVisitTool.from_config(config, compress_llm=compress_llm),
            code_executor=CodeExecutor.from_config(config),
            max_rounds=agent_cfg["max_rounds"],
            explore_max_searches=agent_cfg["explore_max_searches"],
            force_internal=force_internal,
            fetch_top_k=agent_cfg.get("fetch_top_k", 3),
        )

    async def solve(self, question: str, context: str = "",
                    override_skills: Optional[List[Dict]] = None,
                    system_prompt: Optional[str] = None) -> Dict:
        """Solve a question through the multi-round execution flow."""
        clear_page_cache()

        if override_skills is not None:
            relevant_skills = override_skills
        else:
            relevant_skills = self.skill_manager.get_all_skills()
        conversation: List[Dict] = []

        all_search_queries: List[str] = []
        all_search_results: List[Dict] = []
        all_skills_generated: List[Dict] = []

        result = await self._execute(
            question=question,
            context=context,
            skills=relevant_skills,
            conversation=conversation,
            search_queries=all_search_queries,
            search_results=all_search_results,
            skills_generated=all_skills_generated,
            system_prompt=system_prompt,
        )

        profile_stats = result.get("profile_stats", {})
        if profile_stats.get("aborted_on_explore"):
            # Trajectory was aborted at the first explore decision; skip the
            # ensure-answer fallback (which would otherwise burn another LLM
            # call) and return the empty answer as-is.
            answer, answer_with_tag = "", ""
        else:
            ensure_t0 = time.time()
            answer, answer_with_tag = await self._ensure_answer_tag(
                answer_text=result.get("answer", ""),
                conversation=conversation,
                profile_stats=profile_stats,
            )
            profile_stats["ensure_answer_time_sec"] = time.time() - ensure_t0

        actions = result.get("actions", [])

        return {
            "question": question,
            "decision": "EXPLORE" if any(a == "explore" for a in actions) else "INTERNAL",
            "answer": answer,
            "answer_with_tag": answer_with_tag,
            "conversation_history": conversation,
            # No longer separate — exploration is in the main conversation
            "explore_conversation": [],
            "explore_conversations": [],
            "num_rounds": len([m for m in conversation if m["role"] == "assistant"]),
            "skill_used": result.get("skill_used"),
            "skill_generated": all_skills_generated[-1] if all_skills_generated else None,
            "skills_generated_all": all_skills_generated,
            "search_queries": all_search_queries,
            "search_results": all_search_results,
            "actions": actions,
            "called_skill_names": result.get("called_skill_names", []),
            "explore_time_sec": result.get("explore_time_sec", 0.0),
            "profile_stats": profile_stats,
        }

    # ==================== Teacher Skill Rewriting ====================

    async def _rewrite_skill_with_teacher(self, student_skill_data: Dict,
                                           conversation: List[Dict],
                                           exploration_goal: str) -> Optional[Dict]:
        """Rewrite a student-generated skill using the teacher model.

        Extracts all search/page-visit context from the current conversation,
        then asks the teacher model to generate a skill STRICTLY from that
        context (no prior knowledge allowed).
        """
        from core.decision import TEACHER_SKILL_GENERATION_PROMPT
        from json_repair import repair_json

        search_context = self._extract_search_context(conversation)
        if not search_context.strip():
            _log(f"    -> No search context found, skipping teacher rewrite")
            return None

        prompt = TEACHER_SKILL_GENERATION_PROMPT.format(
            exploration_goal=exploration_goal or "Unknown",
            search_context=search_context,
        )

        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self.teacher_llm.chat(messages, temperature=0.3)
            if not response:
                return None
        except Exception as e:
            _log(f"    -> Teacher LLM call failed: {e}")
            return None

        # Try to extract JSON from the response
        # First try the full response, then look for {...} blocks
        for attempt_text in [response, *re.findall(r'\{[\s\S]*\}', response)]:
            try:
                repaired = repair_json(attempt_text)
                parsed = json.loads(repaired)
                if isinstance(parsed, dict) and "skill_name" in parsed:
                    if parsed.get("knowledge") or parsed.get("code"):
                        return parsed
            except (json.JSONDecodeError, Exception):
                continue

        _log(f"    -> Could not parse teacher skill from response")
        return None

    def _extract_search_context(self, conversation: List[Dict]) -> str:
        """Extract search results and page visit content from conversation.

        Collects all user messages that contain search results or
        [Supplementary knowledge] sections, which represent the web
        exploration context the student model had when generating a skill.
        """
        context_parts = []
        for msg in conversation:
            content = msg.get("content", "")
            if not content:
                continue
            # Page visit results contain [Supplementary knowledge]
            if "[Supplementary knowledge]" in content:
                context_parts.append(content)
            # Search results from QUERY
            elif "Search results" in content:
                context_parts.append(content)
        return "\n\n---\n\n".join(context_parts)

    # ==================== Execution Flow ====================

    async def _execute(self, question: str, context: str,
                       skills: List[Dict],
                       conversation: List[Dict],
                       search_queries: List[str],
                       search_results: List[Dict],
                       skills_generated: List[Dict],
                       system_prompt: Optional[str] = None) -> Dict:
        formatted_skills = self.skill_manager.format_skills_for_prompt(skills)
        context_section = f"## Additional Context:\n{context}" if context else ""

        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})

        execution_prompt = get_execution_prompt(
            force_internal=self.force_internal,
            has_skills=bool(skills),
            no_code=(self.code_executor is None),
        )
        first_user_content = execution_prompt.format(
            formatted_skills=formatted_skills,
            question=question,
            context_section=context_section,
        )
        if self.force_first_explore:
            top_banner = (
                "[MANDATORY RULE — REPEATED 1/3]\n"
                "Your FIRST action in this conversation MUST be <EXPLORE>...</EXPLORE>. "
                "<ANSWER>, <ACTIVATE_SKILL>, and <CODE> are FORBIDDEN in round 1.\n\n"
            )
            mid_banner = (
                "\n\n[MANDATORY RULE — REPEATED 2/3]\n"
                "Before you read the question above and react, remember: round 1 MUST "
                "be a single <EXPLORE>...</EXPLORE> tag. Even if the problem looks easy "
                "and you 'know' the answer, you are NOT allowed to skip exploration. "
                "Skipping this step makes your trajectory invalid.\n"
            )
            tail_banner = (
                "\n\n[MANDATORY RULE — REPEATED 3/3]\n"
                "FINAL REMINDER — do not ignore this:\n"
                "  1. Your first action MUST be <EXPLORE>goal</EXPLORE>.\n"
                "  2. Do NOT output <ANSWER>, <ACTIVATE_SKILL>, or <CODE> in round 1.\n"
                "  3. The <EXPLORE> goal must describe the formula, theorem, library "
                "API, algorithm, or fact you want to verify for THIS question.\n"
                "  4. Only after exploration has produced evidence (via <QUERY> and "
                "<PAGE_VISIT>) and a skill has been generated may you answer.\n"
                "Output exactly one <EXPLORE>...</EXPLORE> tag now.\n"
            )
            # Inject middle banner right before the question header so it sits between
            # the rules section and the actual problem statement.
            if "## Question:" in first_user_content:
                first_user_content = first_user_content.replace(
                    "## Question:", mid_banner + "## Question:", 1
                )
            else:
                first_user_content = first_user_content + mid_banner
            first_user_content = top_banner + first_user_content + tail_banner
        conversation.append({
            "role": "user",
            "content": first_user_content,
        })

        latest_skill_used = None
        actions: List[str] = []
        code_context: List[str] = []
        called_skill_names: List[str] = []
        skill_code_retry_count: int = 0
        explore_time_sec: float = 0.0
        profile_stats = {
            "llm_time_sec": 0.0,
            "search_time_sec": 0.0,
            "page_visit_time_sec": 0.0,
            "code_time_sec": 0.0,
            "teacher_rewrite_time_sec": 0.0,
        }
        explore_count: int = 0
        exploration_phase: Optional[str] = None
        raw_explore_intent: str = ""
        exploration_goal: str = ""
        exploration_candidates: List[str] = []

        for round_idx in range(self.max_rounds):
            step_response = await self._chat_with_context_overflow_fallback(
                conversation,
                profile_stats=profile_stats,
            )
            if not step_response:
                break
            if conversation and conversation[-1].get("role") == "assistant" and conversation[-1].get("content") == step_response:
                pass
            else:
                conversation.append({"role": "assistant", "content": step_response})

            step = parse_execution_step(step_response)
            fallback_forced_answer = bool(profile_stats.pop("_context_overflow_fallback_pending", False))
            actions.append(step["type"])
            _log(f"  [Round {round_idx + 1}/{self.max_rounds}] Action: {step['type'].upper()}")
            if fallback_forced_answer and step["type"] != "answer":
                answer_text = strip_answer_tags(step_response).strip()
                _log("    -> Context fallback response was not a valid ANSWER tag; stopping without more tool calls")
                return {
                    "answer": answer_text,
                    "skill_used": latest_skill_used,
                    "actions": actions,
                    "called_skill_names": called_skill_names,
                    "explore_time_sec": explore_time_sec,
                    "profile_stats": profile_stats,
                }
            if step.get("thought"):
                _log(f"    Thought: {step['thought'][:150]}{'...' if len(step.get('thought', '')) > 150 else ''}")

            # --- ANSWER ---
            if step["type"] == "answer":
                if exploration_phase is not None and not self.skip_skill_generation:
                    _log(f"    -> Blocked: must GENERATE_SKILL before ANSWER")
                    conversation.append({
                        "role": "user",
                        "content": (
                            "You are currently in exploration mode. You must first generate a skill "
                            "before you can answer.\n"
                            "Use <QUERY>...</QUERY> for more search, "
                            '<PAGE_VISIT>{"urls": [...], "goal": "..."}</PAGE_VISIT> to visit pages, '
                            "or <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                        ),
                    })
                    continue

                if exploration_phase is not None and self.skip_skill_generation:
                    # In skip_skill_generation mode, allow direct answer from exploration
                    exploration_phase = None
                    raw_explore_intent = ""
                    exploration_goal = ""
                    exploration_candidates = []

                answer_text = step.get("answer", "")
                _log(f"    Answer: {answer_text[:200]}{'...' if len(answer_text) > 200 else ''}")
                return {
                    "answer": answer_text,
                    "skill_used": latest_skill_used,
                    "actions": actions,
                    "called_skill_names": called_skill_names,
                    "explore_time_sec": explore_time_sec,
                    "profile_stats": profile_stats,
                }

            # --- INVALID FORMAT ---
            if step["type"] == "invalid":
                parse_error = step.get("parse_error") or "Invalid action format."
                _log(f"    ParseError: {parse_error}")
                if exploration_phase is not None:
                    if "closing </QUERY>" in parse_error or "closing </GENERATE_SKILL>" in parse_error or "closing </PAGE_VISIT>" in parse_error:
                        phase_error = parse_error
                    elif "Failed to parse <GENERATE_SKILL>" in parse_error or "Failed to parse <PAGE_VISIT>" in parse_error:
                        phase_error = parse_error
                    else:
                        phase_error = "Invalid exploration-step format."
                    if self.skip_skill_generation:
                        feedback_text = (
                            "Format error in the exploration-search step.\n"
                            f"{phase_error}\n"
                            "Use exactly one complete <QUERY>...</QUERY>, "
                            '<PAGE_VISIT>{"urls": [...], "goal": "..."}</PAGE_VISIT>, '
                            "or <ANSWER>...</ANSWER> tag."
                        )
                    else:
                        feedback_text = (
                            "Format error in the exploration-search step.\n"
                            f"{phase_error}\n"
                            "Use exactly one complete <QUERY>...</QUERY>, "
                            '<PAGE_VISIT>{"urls": [...], "goal": "..."}</PAGE_VISIT>, '
                            "or <GENERATE_SKILL>{...}</GENERATE_SKILL> tag."
                        )
                else:
                    no_code = (self.code_executor is None)
                    if self.force_internal and no_code and not skills:
                        valid_actions = "Use <ANSWER>...</ANSWER>."
                    elif self.force_internal and no_code:
                        valid_actions = (
                            "Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL> "
                            "or <ANSWER>...</ANSWER>."
                        )
                    elif self.force_internal and not skills:
                        valid_actions = (
                            "Use <CODE>...code...</CODE> "
                            "or <ANSWER>...</ANSWER>."
                        )
                    elif self.force_internal:
                        valid_actions = (
                            "Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, "
                            "<CODE>...code...</CODE>, "
                            "or <ANSWER>...</ANSWER>."
                        )
                    else:
                        valid_actions = (
                            "Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, "
                            "<CODE>...code...</CODE>, "
                            "<ANSWER>...</ANSWER>, "
                            "or <EXPLORE>...</EXPLORE>."
                        )
                    feedback_text = (
                        "Format error.\n"
                        f"{parse_error}\n"
                        f"{valid_actions}"
                    )
                conversation.append({
                    "role": "user",
                    "content": feedback_text,
                })
                continue

            # --- EXPLORE ---
            if step["type"] == "explore":
                # Early abort: callers (e.g. direct-only data collection) can
                # opt-in to stop the trajectory the moment the model decides
                # to explore, avoiding the cost of running the full explore
                # sub-flow only to discard the result later.
                if self.abort_on_explore:
                    _log(f"    -> Aborted: abort_on_explore=True")
                    profile_stats["aborted_on_explore"] = True
                    break
                missing_capability = step.get("explore_intent", "")
                _log(f"    Explore intent: {missing_capability}")

                if self.force_internal:
                    _log(f"    -> Blocked by --force_internal")
                    no_code = (self.code_executor is None)
                    if no_code and skills:
                        hint = ("Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL> "
                                "or <ANSWER>...</ANSWER>.")
                    elif no_code:
                        hint = "Use <ANSWER>...</ANSWER>."
                    elif skills:
                        hint = ("Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, "
                                "<CODE>...code...</CODE>, or <ANSWER>...</ANSWER>.")
                    else:
                        hint = "Use <CODE>...code...</CODE> or <ANSWER>...</ANSWER>."
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Exploration is disabled. Please answer using your existing "
                            f"knowledge.\n{hint}"
                        ),
                    })
                    continue

                if exploration_phase is not None:
                    _log(f"    -> Blocked: already in exploration mode")
                    conversation.append({
                        "role": "user",
                        "content": (
                            "You have already entered exploration mode.\n"
                            "Use <QUERY>...</QUERY> to continue searching, "
                            '<PAGE_VISIT>{"urls": [...], "goal": "..."}</PAGE_VISIT> to visit pages, '
                            "or <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                        ),
                    })
                    continue

                # Skip the NEW_GOAL rewriting step — go directly to search phase
                # using the raw explore intent as the exploration goal.
                exploration_phase = "search"
                raw_explore_intent = missing_capability
                exploration_goal = missing_capability
                exploration_candidates = []
                seen = set()
                for skill_name in called_skill_names:
                    if skill_name and skill_name not in seen:
                        exploration_candidates.append(skill_name)
                        seen.add(skill_name)
                for skill in skills:
                    normalized = normalize_skill(skill)
                    if normalized.get("builtin"):
                        continue
                    skill_name = normalized.get("skill_name", "")
                    if skill_name and skill_name not in seen:
                        exploration_candidates.append(skill_name)
                        seen.add(skill_name)

                conversation.append({
                    "role": "user",
                    "content": get_execution_prompt(
                        force_internal=self.force_internal,
                        mode="explore_search_no_skill" if self.skip_skill_generation else "explore_search",
                    ).format(
                        exploration_goal=exploration_goal or "(empty)",
                    ),
                })
                continue

            # --- QUERY ---
            if step["type"] == "query":
                queries = step.get("queries", [])
                if queries:
                    _log(f"    Search queries: {queries}")
                else:
                    _log(f"    Search queries: (empty)")

                if exploration_phase is None:
                    _log(f"    -> Blocked: QUERY without prior EXPLORE")
                    conversation.append({
                        "role": "user",
                        "content": (
                            "You are not in exploration mode yet. "
                            "Use <EXPLORE>...</EXPLORE> first if you need exploration."
                        ),
                    })
                    continue

                if explore_count >= self.explore_max_searches:
                    _log(f"    -> Explore limit reached ({self.explore_max_searches})")
                    if self.skip_skill_generation:
                        conversation.append({
                            "role": "user",
                            "content": (
                                f"Search limit reached ({self.explore_max_searches} searches).\n"
                                "Use the information gathered so far to answer the question directly.\n"
                                "Use <ANSWER>...</ANSWER>."
                            ),
                        })
                    else:
                        conversation.append({
                            "role": "user",
                            "content": (
                                f"Search limit reached ({self.explore_max_searches} searches).\n"
                                "You must now generate a reusable skill from the information gathered so far.\n"
                                "Use <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                            ),
                        })
                    continue

                if not queries:
                    conversation.append({
                        "role": "user",
                        "content": 'No queries provided. Use <QUERY>["search keyword"]</QUERY>.',
                    })
                    continue

                t0 = time.time()
                search_data = await self.explorer.execute_search(queries, knowledge_gaps=exploration_goal)
                search_elapsed = time.time() - t0
                explore_time_sec += search_elapsed
                profile_stats["search_time_sec"] += search_elapsed
                if search_data.get("search_results"):
                    explore_count += 1

                search_queries.extend(queries)
                search_results.extend(search_data.get("search_results", []))

                formatted_info = search_data.get("formatted_info", "No results.")
                if len(formatted_info) > 4000:
                    formatted_info = formatted_info[:4000] + "\n...[truncated]"

                queries_str = "; ".join(queries)
                if self.skip_skill_generation:
                    next_actions_hint = (
                        "IMPORTANT: Search snippets above are brief and may be inaccurate. "
                        "You SHOULD visit the most relevant pages to read their full content before answering.\n"
                        'Next step: Use <PAGE_VISIT>{"urls": ["url1", ...], "goal": "what to extract"}</PAGE_VISIT> '
                        "to read page details.\n"
                        "Or <QUERY>...</QUERY> to search more, or <ANSWER>...</ANSWER> if you already have enough detail."
                    )
                else:
                    next_actions_hint = (
                        "IMPORTANT: Search snippets above are brief and may be inaccurate. "
                        "You SHOULD visit the most relevant pages to read their full content before generating a skill.\n"
                        'Next step: Use <PAGE_VISIT>{"urls": ["url1", ...], "goal": "what to extract"}</PAGE_VISIT> '
                        "to read page details.\n"
                        "Or <QUERY>...</QUERY> to search more, "
                        "or <GENERATE_SKILL>{...}</GENERATE_SKILL> if you already have enough detail.\n"
                        "If you are extending an existing skill, reuse its exact skill_name."
                    )
                conversation.append({
                    "role": "user",
                    "content": (
                        f"Exploration goal: {exploration_goal}\n"
                        f"Search results for queries: {queries_str}\n\n"
                        f"{formatted_info}\n\n"
                        f"{next_actions_hint}"
                    ),
                })
                continue

            # --- PAGE_VISIT ---
            if step["type"] == "page_visit":
                page_visit_data = step.get("page_visit") or {}
                urls = page_visit_data.get("urls", [])
                goal = page_visit_data.get("goal", exploration_goal or "")
                _log(f"    Page visit: {len(urls)} URL(s), goal={goal[:80]}")

                if exploration_phase is None:
                    _log(f"    -> Blocked: PAGE_VISIT without prior EXPLORE")
                    conversation.append({
                        "role": "user",
                        "content": (
                            "You are not in exploration mode yet. "
                            "Use <EXPLORE>...</EXPLORE> first if you need exploration."
                        ),
                    })
                    continue

                if not urls:
                    conversation.append({
                        "role": "user",
                        "content": (
                            'No URLs provided. Use <PAGE_VISIT>{"urls": ["url1", "url2"], "goal": "what to extract"}</PAGE_VISIT>.'
                        ),
                    })
                    continue

                t0 = time.time()
                visit_data = await self.explorer.execute_page_visit(urls, goal=goal)
                visit_elapsed = time.time() - t0
                explore_time_sec += visit_elapsed
                profile_stats["page_visit_time_sec"] += visit_elapsed

                # Track page visit success/failure for eval monitoring
                global _page_visit_success, _page_visit_failed, _page_visit_forbidden
                with _page_visit_lock:
                    for vr in visit_data.get("visited_pages", []):
                        src = vr.get("fetch_source")
                        if src == "failed":
                            _page_visit_failed += 1
                        elif src == "forbidden":
                            _page_visit_forbidden += 1
                        else:
                            _page_visit_success += 1

                formatted_info = visit_data.get("formatted_info", "No content extracted.")
                if len(formatted_info) > 4000:
                    formatted_info = formatted_info[:4000] + "\n...[truncated]"

                if self.skip_skill_generation:
                    page_next_actions_hint = (
                        "You now have detailed page content. Use it to answer the question.\n"
                        "Use <ANSWER>...</ANSWER> to provide your final answer.\n"
                        "Or <PAGE_VISIT>...</PAGE_VISIT> / <QUERY>...</QUERY> if you need more information."
                    )
                else:
                    page_next_actions_hint = (
                        "You now have detailed page content with [Supplementary knowledge]. "
                        "Synthesize this into a reusable skill.\n"
                        "Use <GENERATE_SKILL>{...}</GENERATE_SKILL> to generate a skill.\n"
                        "Or <PAGE_VISIT>...</PAGE_VISIT> / <QUERY>...</QUERY> if you need more information.\n"
                        "If you are extending an existing skill, reuse its exact skill_name."
                    )

                conversation.append({
                    "role": "user",
                    "content": (
                        f"Page visit results for {len(urls)} URL(s):\n\n"
                        f"{formatted_info}\n\n"
                        f"{page_next_actions_hint}"
                    ),
                })
                continue

            # --- GENERATE_SKILL ---
            if step["type"] == "generate_skill":
                if self.skip_skill_generation:
                    _log(f"    -> Blocked: skill generation disabled in this mode")
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Skill generation is disabled in this mode. "
                            "Use the information you have gathered to answer directly.\n"
                            "Use <ANSWER>...</ANSWER> to provide your answer, "
                            "or <QUERY>...</QUERY> / <PAGE_VISIT>...</PAGE_VISIT> to gather more information."
                        ),
                    })
                    continue

                if exploration_phase is None:
                    _log(f"    -> Blocked: GENERATE_SKILL without prior EXPLORE")
                    no_code = (self.code_executor is None)
                    if self.force_internal and no_code and not skills:
                        hint = "Use <ANSWER>...</ANSWER>."
                    elif self.force_internal and no_code:
                        hint = ("Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL> "
                                "or <ANSWER>...</ANSWER>.")
                    elif self.force_internal and not skills:
                        hint = "Use <CODE>...code...</CODE> or <ANSWER>...</ANSWER>."
                    elif self.force_internal:
                        hint = ("Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, "
                                "<CODE>...code...</CODE>, or <ANSWER>...</ANSWER>.")
                    else:
                        hint = ("Use <ACTIVATE_SKILL>, <CODE>, <ANSWER>, or <EXPLORE>.")
                    conversation.append({
                        "role": "user",
                        "content": (
                            "You haven't explored yet. "
                            + (
                                f"Exploration is disabled in this run. {hint}"
                                if self.force_internal
                                else "Use <EXPLORE>...</EXPLORE> to enter exploration first, "
                                     f"then search and generate a skill.\nOr {hint}"
                            )
                        ),
                    })
                    continue

                if explore_count == 0:
                    _log(f"    -> Blocked: GENERATE_SKILL without any successful search")
                    conversation.append({
                        "role": "user",
                        "content": (
                            "You must perform at least one search before generating a skill. "
                            'Use <QUERY>["search keyword"]</QUERY> to search for information first.'
                        ),
                    })
                    continue

                skill_data = step.get("skill_data")
                if skill_data:
                    skill_reason = str(skill_data.get("reason", "")).strip()
                    if skill_reason:
                        _log(f"    Skill reason: {skill_reason[:200]}")

                # --- Teacher skill rewriting ---
                teacher_rewritten = False
                if self.teacher_llm is not None and skill_data:
                    _log(f"    -> Teacher skill rewriting: calling teacher model...")
                    teacher_t0 = time.time()
                    teacher_skill = await self._rewrite_skill_with_teacher(
                        skill_data, conversation,
                        exploration_goal or raw_explore_intent,
                    )
                    profile_stats["teacher_rewrite_time_sec"] += time.time() - teacher_t0
                    if teacher_skill is not None:
                        old_name = skill_data.get("skill_name", "?")
                        new_name = teacher_skill.get("skill_name", "?")
                        _log(f"    -> Teacher rewrote skill: {old_name} -> {new_name}")
                        skill_data = teacher_skill
                        teacher_rewritten = True
                    else:
                        _log(f"    -> Teacher rewrite failed, keeping student's skill")

                new_skill = (
                    self.explorer.parse_and_register_skill(
                        skill_data,
                        current_skills=skills,
                        preferred_skill_names=exploration_candidates,
                    )
                    if skill_data else None
                )

                if not new_skill:
                    _log(f"    -> Failed to parse/register skill")
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Failed to parse the skill. Please provide valid JSON with at least "
                            '"skill_name" and one of "workflow" or "code".\n'
                            "Use <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                        ),
                    })
                    continue

                new_skill = normalize_skill(new_skill)
                code = skill_code(new_skill)

                # Validate skill code syntax before injection
                skill_code_syntax_error = None
                if code and code.strip():
                    try:
                        compile(code.strip(), "<skill>", "exec")
                    except SyntaxError as e:
                        skill_code_syntax_error = e

                if skill_code_syntax_error and skill_code_retry_count < 2:
                    skill_code_retry_count += 1
                    _log(f"    -> Skill code SyntaxError (attempt {skill_code_retry_count}/2): "
                         f"{skill_code_syntax_error}")
                    conversation.append({
                        "role": "user",
                        "content": (
                            f"Skill code has a syntax error: {skill_code_syntax_error}\n\n"
                            "Please regenerate the complete skill with fixed code. "
                            "Make sure all parentheses, brackets, and quotes are properly closed.\n"
                            "Use <GENERATE_SKILL>{...}</GENERATE_SKILL> with the corrected code."
                        ),
                    })
                    continue

                # Either code is valid, no code, or max retries exhausted
                skill_code_valid = True
                if skill_code_syntax_error:
                    skill_code_valid = False
                    _log(f"    -> Skill code SyntaxError persists after {skill_code_retry_count} "
                         f"retries, registering workflow only")
                _exhausted_retries = skill_code_retry_count
                skill_code_retry_count = 0

                updated = False
                if new_skill.get("action") == "evolve" and new_skill.get("skill_id"):
                    for idx, existing in enumerate(skills):
                        if normalize_skill(existing).get("skill_id") == new_skill["skill_id"]:
                            skills[idx] = new_skill
                            updated = True
                            break
                if not updated and new_skill.get("action") == "evolve":
                    target_name = self._normalize_skill_name(new_skill.get("skill_name", ""))
                    for idx, existing in enumerate(skills):
                        if self._normalize_skill_name(normalize_skill(existing).get("skill_name", "")) == target_name:
                            skills[idx] = new_skill
                            updated = True
                            break
                if not updated:
                    target_name = self._normalize_skill_name(new_skill.get("skill_name", ""))
                    name_conflict = False
                    if target_name:
                        for idx, existing in enumerate(skills):
                            if self._normalize_skill_name(normalize_skill(existing).get("skill_name", "")) == target_name:
                                skills[idx] = new_skill
                                name_conflict = True
                                break
                    if not name_conflict:
                        recalled_ids = {normalize_skill(s).get("skill_id") for s in skills}
                        if new_skill.get("skill_id") and new_skill["skill_id"] not in recalled_ids:
                            skills.append(new_skill)

                skills_generated.append(new_skill)
                new_skill["source_question"] = question
                if teacher_rewritten:
                    new_skill["_teacher_rewritten"] = True

                # Auto-inject code into code_context only if syntax is valid
                if skill_code_valid and self.code_executor is not None and code and code.strip() and code.strip() not in code_context:
                    code_context.append(code.strip())

                # Track as activated
                latest_skill_used = new_skill.get("skill_id") or latest_skill_used
                activated_name = self._normalize_skill_name(new_skill.get("skill_name", ""))
                if activated_name and activated_name not in called_skill_names:
                    called_skill_names.append(activated_name)

                # Build feedback
                skill_name = new_skill.get("skill_name", "Unknown")
                workflow = skill_workflow(new_skill)

                blocks = [f"### {skill_name}"]
                if workflow:
                    blocks.append("Workflow:\n" + workflow)
                if code and skill_code_valid:
                    if self.code_executor is not None:
                        blocks.append(
                            "Code (loaded and available for <CODE>):\n\n"
                            f"```python\n{code}\n```"
                        )
                    else:
                        blocks.append(
                            "Reference code:\n\n"
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

                if self.code_executor is not None:
                    activation_hint = (
                        "\n\nThis skill is already activated — workflow is loaded and code is available for <CODE>. "
                        "No need to call <ACTIVATE_SKILL> for it again.\n"
                    )
                else:
                    activation_hint = (
                        "\n\nThis skill is already activated — workflow is loaded. "
                        "Code is provided as reference only (no code execution available). "
                        "No need to call <ACTIVATE_SKILL> for it again.\n"
                    )

                conversation.append({
                    "role": "user",
                    "content": (
                        (
                            "Skill evolved and registered (auto-activated):\n\n"
                            if new_skill.get("action") == "evolve"
                            else "Skill generated and registered (auto-activated):\n\n"
                        )
                        + "\n\n".join(blocks)
                        + activation_hint
                        + self._next_step_actions_hint(code_context, has_skills=bool(skills))
                    ),
                })
                continue

            # --- ACTIVATE_SKILL (parsed as type="skill_call") ---
            if step["type"] == "skill_call":
                if exploration_phase is not None:
                    if self.skip_skill_generation:
                        _log(f"    -> Blocked: ACTIVATE_SKILL during exploration, answer directly")
                        conversation.append({
                            "role": "user",
                            "content": (
                                "You are currently in exploration mode. "
                                "Use the information gathered to answer directly.\n"
                                "Use <QUERY>...</QUERY> for more search, "
                                '<PAGE_VISIT>{"urls": [...], "goal": "..."}</PAGE_VISIT> to visit pages, '
                                "or <ANSWER>...</ANSWER> to answer."
                            ),
                        })
                    else:
                        _log(f"    -> Blocked: must GENERATE_SKILL before ACTIVATE_SKILL")
                        conversation.append({
                            "role": "user",
                            "content": (
                                "You are currently in exploration mode. You must first generate a skill "
                                "before activating skills.\n"
                                "Use <QUERY>...</QUERY> for more search, "
                                '<PAGE_VISIT>{"urls": [...], "goal": "..."}</PAGE_VISIT> to visit pages, '
                                "or <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                            ),
                        })
                    continue

                call_info = step.get("skill_call") or {}
                skill_name = call_info.get("skill_name", "")
                _log(f"    Skill: {skill_name}")
                step_result = self._execute_skill_call(step, skills, code_context)
                latest_skill_used = step_result.get("skill_used") or latest_skill_used
                used_skill_name = self._normalize_skill_name(skill_name)
                if (
                    step_result.get("skill_used")
                    and used_skill_name
                ):
                    called_skill_names.append(used_skill_name)

                feedback = step_result.get("feedback", "This round completed.")
                if len(feedback) > 2500:
                    feedback = feedback[:2500] + "..."
                conversation.append({
                    "role": "user",
                    "content": (
                        f"Execution feedback:\n{feedback}\n\n"
                        "Continue with your next step.\n"
                        + self._next_step_actions_hint(code_context, has_skills=bool(skills))
                    ),
                })
                continue

            # --- CODE (standalone code execution) ---
            if step["type"] == "run_code":
                if exploration_phase is not None:
                    _log(f"    -> Blocked: CODE during exploration")
                    conversation.append({
                        "role": "user",
                        "content": (
                            "You are currently in exploration mode. "
                            "Focus on gathering information.\n"
                            "Use <QUERY>...</QUERY> for more search, "
                            '<PAGE_VISIT>{"urls": [...], "goal": "..."}</PAGE_VISIT> to visit pages, '
                            + (
                                "or <ANSWER>...</ANSWER> to answer."
                                if self.skip_skill_generation
                                else "or <GENERATE_SKILL>{...}</GENERATE_SKILL>."
                            )
                        ),
                    })
                    continue

                if self.code_executor is None:
                    _log(f"    -> Blocked: code_executor is None")
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Code execution is not available.\n"
                            + self._next_step_actions_hint(code_context, has_skills=bool(skills))
                        ),
                    })
                    continue

                code = step.get("run_code", "")
                _log(f"    Run code: {code[:200]}{'...' if len(code) > 200 else ''}")
                code_t0 = time.time()
                feedback = self._execute_run_code(code, code_context)
                profile_stats["code_time_sec"] += time.time() - code_t0
                if len(feedback) > 2500:
                    feedback = feedback[:2500] + "..."
                conversation.append({
                    "role": "user",
                    "content": (
                        f"Code execution feedback:\n{feedback}\n\n"
                        "Continue with your next step.\n"
                        + self._next_step_actions_hint(code_context, has_skills=bool(skills))
                    ),
                })
                continue

        # Max rounds reached without <ANSWER>, force one
        return {
            "answer": "",
            "skill_used": latest_skill_used,
            "actions": actions,
            "called_skill_names": called_skill_names,
            "explore_time_sec": explore_time_sec,
            "profile_stats": profile_stats,
        }

    # ==================== Skill Execution ====================

    def _execute_skill_call(self, step: Dict, skills: List[Dict],
                            code_context: Optional[List[str]] = None) -> Dict:
        """Execute a skill activation based on parsed plan.

        Only handles domain skills (knowledge/code). Code execution is
        handled separately by _execute_run_code().
        """
        call = step.get("skill_call") or {}
        skill_name = call.get("skill_name", "")
        matched_skill = self._find_skill_by_name(skill_name, skills)
        if code_context is None:
            code_context = []

        if not matched_skill:
            available = [
                normalize_skill(s).get("skill_name", "")
                for s in skills
                if isinstance(s, dict)
            ]
            return {
                "feedback": (
                    f"Skill '{skill_name}' not found. "
                    f"Available skill_name values: {available}. Please try again."
                ),
                "skill_used": None,
            }

        matched_skill = normalize_skill(matched_skill)
        return self._execute_regular_skill(matched_skill, code_context)

    def _execute_regular_skill(self, skill: Dict, code_context: List[str]) -> Dict:
        """Execute a domain skill that may contain workflow, code, or both."""
        workflow = skill_workflow(skill)
        code = skill_code(skill).strip()
        if code:
            try:
                compile(code, "<skill>", "exec")
            except SyntaxError as e:
                _log(f"    -> SyntaxError in activated skill code, skipping: {e}")
                code = ""
        if not workflow and not code:
            return {
                "feedback": "This skill has neither workflow nor code content.",
                "skill_used": skill.get("skill_id"),
            }

        sections = []
        if workflow:
            if self.skill_reference_only:
                sections.append(
                    f"Reference workflow from skill '{skill['skill_name']}' "
                    "(treat as guidance only; adapt it to the current problem rather than "
                    f"following it literally):\n\n{workflow}"
                )
            else:
                sections.append(f"Workflow from skill '{skill['skill_name']}':\n\n{workflow}")

        if code:
            if self.skill_reference_only:
                sections.append(
                    f"Reference code from skill '{skill['skill_name']}' "
                    "(for inspiration ONLY — it is NOT loaded into the execution environment and "
                    "may be too specific to the original problem; write your own adapted code in "
                    f"<CODE>...</CODE> based on this reference):\n\n```python\n{code}\n```"
                )
            elif self.code_executor is not None:
                if code not in code_context:
                    code_context.append(code)
                sections.append(
                    f"Code from skill '{skill['skill_name']}' loaded. The following function is now available:\n\n"
                    f"```python\n{code}\n```\n\n"
                    "Use <CODE> to call the function directly. No import is needed."
                )
            else:
                sections.append(
                    f"Reference code from skill '{skill['skill_name']}':\n\n"
                    f"```python\n{code}\n```"
                )

        if not skill.get("builtin") and skill.get("skill_id"):
            self.skill_manager.update_stats(skill["skill_id"], success=True)

        return {"feedback": "\n\n".join(sections), "skill_used": skill.get("skill_id")}

    def _execute_run_code(self, code: str, code_context: List[str]) -> str:
        """Execute Python code with accumulated code_context prepended.

        Returns a feedback string (not a Dict) for simpler caller code.
        """
        if not code:
            return (
                "No code provided. Put your Python code inside <CODE>...</CODE>."
            )

        # Fix double-escaped newlines from small models.
        # e.g. JSON "code": "a=1\\nb=2" → json.loads → literal \n instead of real newline.
        # Detection: if code has literal \n but no real newlines, unescape them.
        if "\n" not in code and "\\n" in code:
            code = code.replace("\\n", "\n").replace("\\t", "\t")

        if code_context:
            preamble = "\n\n".join(code_context)
            full_code = f"{preamble}\n\n# --- User code ---\n{code}"
        else:
            full_code = code

        exec_result = self.code_executor.execute(full_code)

        if exec_result["success"]:
            stdout = exec_result["stdout"].strip()
            if stdout:
                return f"Code executed successfully.\nOutput:\n{stdout}"
            else:
                return (
                    "Code executed successfully, but produced no output.\n"
                    "Make sure to use print() to output the results you want to see. "
                    "For example, instead of just computing a value, write print(result) to display it."
                )
        else:
            # Layer 2: if skill code caused the failure, remove it and retry
            if code_context:
                retry_result = self.code_executor.execute(code)
                if retry_result["success"]:
                    _log(f"    -> Skill code caused execution failure, "
                         f"removed {len(code_context)} skill code block(s)")
                    code_context.clear()
                    stdout = retry_result["stdout"].strip()
                    if stdout:
                        return (
                            "Note: Previously loaded skill code had errors and was removed.\n"
                            f"Code executed successfully.\nOutput:\n{stdout}"
                        )
                    else:
                        return (
                            "Note: Previously loaded skill code had errors and was removed.\n"
                            "Code executed successfully, but produced no output.\n"
                            "Make sure to use print() to output the results."
                        )
            return (
                f"Code execution failed.\n"
                f"stdout: {exec_result['stdout']}\n"
                f"stderr: {exec_result['stderr']}"
            )

    def _verify_code_str(self, code: str) -> Optional[str]:
        """Verify that a raw code string can be executed cleanly.

        Returns None if code is empty or execution succeeds.
        Returns an error message string if execution fails.
        """
        if not code or not code.strip():
            return None
        code = code.strip()
        # Fix double-escaped newlines from small models
        if "\n" not in code and "\\n" in code:
            code = code.replace("\\n", "\n").replace("\\t", "\t")
        try:
            result = self.code_executor.execute(code)
            if result["success"]:
                return None
            return (
                f"Error: {result['stderr'].strip() or 'Unknown execution error'}\n"
                f"(stdout: {result['stdout'].strip() or 'none'})"
            )
        except Exception as e:
            return f"Verification crashed: {e}"

    def _verify_skill_code(self, skill: Dict) -> Optional[str]:
        """Verify that a skill's code can be executed cleanly.

        Returns None if the skill has no code or execution succeeds.
        Returns an error message string if execution fails.
        """
        from core.skill_schema import skill_code
        return self._verify_code_str(skill_code(skill))

    # ==================== Helpers ====================

    async def _chat_with_context_overflow_fallback(self,
                                                   conversation: List[Dict],
                                                   profile_stats: Optional[Dict] = None) -> Optional[str]:
        llm_t0 = time.time()
        response = await self.llm.chat(conversation)
        if profile_stats is not None:
            profile_stats["llm_time_sec"] = profile_stats.get("llm_time_sec", 0.0) + (time.time() - llm_t0)
        if response is not None:
            return response
        if not self._should_try_context_overflow_fallback(conversation):
            return response

        original_chars = sum(len(str(msg.get("content", ""))) for msg in conversation)
        original_assistant_chars = sum(
            len(str(msg.get("content", "")))
            for msg in conversation
            if msg.get("role") == "assistant"
        )
        if profile_stats is not None:
            profile_stats["context_overflow_fallback_attempted"] = True
            profile_stats["context_overflow_fallback_error"] = getattr(self.llm, "last_error_message", "")
        print(
            "[ContextFallback] primary_failed "
            f"error_type={getattr(self.llm, 'last_error_type', None)} "
            f"messages={len(conversation)} chars={original_chars} "
            f"assistant_chars={original_assistant_chars}"
        )
        direct_prompt = self._build_direct_fallback_prompt(conversation)
        direct_chars = sum(len(str(msg.get("content", ""))) for msg in direct_prompt)
        if profile_stats is not None:
            profile_stats["context_overflow_fallback_strategy"] = "direct_question_only"
        print(
            "[ContextFallback] retry_direct_question_only "
            f"messages={len(direct_prompt)} chars={direct_chars} "
            f"removed_chars={original_chars - direct_chars}"
        )
        llm_t0 = time.time()
        response = await self.llm.chat(direct_prompt)
        if profile_stats is not None:
            profile_stats["llm_time_sec"] = profile_stats.get("llm_time_sec", 0.0) + (time.time() - llm_t0)
        if response is None:
            if profile_stats is not None:
                profile_stats["context_overflow_fallback_failed"] = True
                profile_stats["context_overflow_fallback_retry_error"] = getattr(self.llm, "last_error_message", "")
            print(
                "[ContextFallback] fallback_failed "
                f"error_type={getattr(self.llm, 'last_error_type', None)} "
                f"error={getattr(self.llm, 'last_error_message', '')}"
            )
            return response
        if profile_stats is not None:
            profile_stats["context_overflow_fallback"] = True
            profile_stats["context_overflow_fallback_succeeded"] = True
            profile_stats["_context_overflow_fallback_pending"] = True
        print(f"[ContextFallback] fallback_succeeded response_chars={len(response)}")
        conversation.append({
            "role": "user",
            "content": "Context overflow fallback: answered directly from the original question only.",
        })
        conversation.append({"role": "assistant", "content": response})
        return response

    def _should_try_context_overflow_fallback(self, conversation: List[Dict]) -> bool:
        if getattr(self.llm, "last_error_type", None) == "context_overflow":
            return True
        total_chars = sum(len(str(msg.get("content", ""))) for msg in conversation)
        return total_chars >= 80000

    def _build_direct_fallback_prompt(self, conversation: List[Dict]) -> List[Dict]:
        question = ""
        for msg in conversation:
            if msg.get("role") != "user":
                continue
            content = str(msg.get("content", ""))
            match = re.search(r"## Question:\s*(.*?)(?:\n\nRules:|\nRules:|\Z)", content, flags=re.DOTALL)
            if match:
                question = match.group(1).strip()
                break
        if not question:
            for msg in conversation:
                if msg.get("role") == "user":
                    question = str(msg.get("content", "")).strip()
                    break
        return [{
            "role": "user",
            "content": (
                "Please answer the following question. Think step by step, then put your final answer inside <ANSWER></ANSWER> tags.\n\n"
                "For multiple-choice questions, put only the option letter inside the tags, e.g. <ANSWER>B</ANSWER>.\n"
                "For open-ended questions, put a concise answer inside the tags.\n\n"
                f"{question}"
            ),
        }]

    def _strip_think_from_conversation(self, conversation: List[Dict]) -> List[Dict]:
        compact = []
        for msg in conversation:
            copied = dict(msg)
            if copied.get("role") == "assistant":
                copied["content"] = re.sub(
                    r"<think\b[^>]*>.*?</think>",
                    "",
                    str(copied.get("content", "")),
                    flags=re.IGNORECASE | re.DOTALL,
                ).strip()
            compact.append(copied)
        return compact

    async def _ensure_answer_tag(self, answer_text: str,
                                 conversation: List[Dict],
                                 profile_stats: Optional[Dict] = None) -> tuple[str, str]:
        for msg in reversed(conversation):
            if msg.get("role") != "assistant":
                continue
            extracted = extract_answer_from_tag(msg.get("content", ""))
            if extracted:
                return extracted, f"<ANSWER>{extracted}</ANSWER>"

        extracted = extract_answer_from_tag(answer_text)
        if extracted:
            return extracted, f"<ANSWER>{extracted}</ANSWER>"

        conversation.append({"role": "user", "content": FINAL_ANSWER_PROMPT})
        response = await self._chat_with_context_overflow_fallback(
            conversation,
            profile_stats=profile_stats,
        )
        if response:
            if not (conversation and conversation[-1].get("role") == "assistant" and conversation[-1].get("content") == response):
                conversation.append({"role": "assistant", "content": response})
            extracted = extract_answer_from_tag(response)
            if extracted:
                return extracted, f"<ANSWER>{extracted}</ANSWER>"

        fallback = strip_answer_tags(response or answer_text).strip()
        return fallback, f"<ANSWER>{fallback}</ANSWER>"

    def _find_skill_by_name(self, name: Optional[str], skills: List[Dict]) -> Optional[Dict]:
        if not skills or not name:
            return None
        target_skill_name = self._normalize_skill_name(name)
        if not target_skill_name:
            return None
        for s in skills:
            normalized = normalize_skill(s)
            skill_name = self._normalize_skill_name(normalized.get("skill_name", ""))
            if skill_name and skill_name == target_skill_name:
                return normalized
        return None

    def _normalize_skill_name(self, value: Optional[str]) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text

    def _next_step_actions_hint(self, code_context: Optional[List[str]],
                                has_skills: bool = True) -> str:
        no_code = (self.code_executor is None)
        if no_code and self.force_internal and not has_skills:
            base = "Use <ANSWER>...</ANSWER>."
        elif no_code and self.force_internal:
            base = ("Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL> "
                    "or <ANSWER>...</ANSWER>.")
        elif no_code:
            base = ("Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, "
                    "<ANSWER>...</ANSWER>, or <EXPLORE>...</EXPLORE>.")
        elif self.force_internal and not has_skills:
            base = "Use <CODE>...code...</CODE> or <ANSWER>...</ANSWER>."
        elif self.force_internal:
            base = ("Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, <CODE>...code...</CODE>, "
                    "or <ANSWER>...</ANSWER>.")
        else:
            base = ("Use <ACTIVATE_SKILL>skill_name</ACTIVATE_SKILL>, <CODE>...code...</CODE>, "
                    "<ANSWER>...</ANSWER>, or <EXPLORE>...</EXPLORE>.")
        if code_context and not no_code and not self.skill_reference_only:
            base += (
                "\nYou can use <CODE> to directly call functions loaded from skill code. "
                "No import is needed."
            )
        if self.skill_reference_only and not no_code:
            base += (
                "\nNote: skill code shown above is reference only and is NOT loaded. "
                "Write your own adapted code in <CODE>...</CODE> based on the reference."
            )
        return base
