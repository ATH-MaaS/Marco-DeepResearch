"""EvolveR-style memory agent: FreeToolAgent + auto-injected experience principles.

Based on free_tool_agent.py with the following additions:
  - Before each question, retrieve relevant principles from ExperienceManager
    and inject them into the prompt (auto-inject, no new model action needed)
  - After each question, record the trajectory for later distillation
  - Supports the full EvolveR lifecycle when used with eval_evolve_memory mode

The agent itself does NOT modify the memory — that's the eval loop's job
(distillation, metric update, pruning happen between questions).
"""

import time
import threading
from typing import Dict, List, Optional, Tuple

from core.decision import parse_execution_step
from core.explorer import Explorer
from core.experience_manager import ExperienceManager, ExperiencePrinciple
from tools.code_executor import CodeExecutor
from tools.web_search import WebSearchTool
from tools.web_visit import WebVisitTool
from llm.client import LLMClient


EVOLVE_MEMORY_PROMPT = """\
You are an intelligent assistant solving problems in multiple rounds.
You have direct access to web search, page reading, and Python code execution.
{experience_section}
## Question:
{question}
{context_section}

Rules:
- Think step-by-step in plain text, then output EXACTLY ONE action tag at the end.
- You will receive feedback after each action.
- Never output raw code directly; always use <CODE> to run Python code.
- CRITICAL: Do NOT guess when uncertain. If you are not highly confident (>90%) in your answer, use <QUERY> to search for evidence first.

Uncertainty signals (any of these means you should search):
- The question asks about specific facts, names, dates, numbers, or formulas you are not 100% sure about
- You find yourself torn between two or more options
- The topic is highly specialized or obscure
- You are relying on vague memory rather than solid knowledge

Actions (Choose exactly one):
1) Search web (at most 3 queries per call): <QUERY>["query 1", "query 2"]</QUERY>
2) Visit pages to read full content (at most 4 URLs per call): <PAGE_VISIT>{{"urls": ["url1", "url2"], "goal": "what to extract"}}</PAGE_VISIT>
3) Run Python code: <CODE>
   your python code here
   </CODE>
4) Answer — ONLY when confident: <ANSWER>final answer (or option letter like B for multiple choice)</ANSWER>

Note: <QUERY> supports at most 3 queries per call; <PAGE_VISIT> supports at most 4 URLs per call. Excess entries will be silently dropped.
IMPORTANT: Search snippets from <QUERY> are brief summaries. Always use <PAGE_VISIT> to read full page content before answering if the snippet lacks detail."""


# Page-visit tracking
_page_visit_success = 0
_page_visit_failed = 0
_page_visit_forbidden = 0
_page_visit_lock = threading.Lock()


class EvolveMemoryAgent:
    """FreeToolAgent + EvolveR-style auto-injected experience principles."""

    def __init__(self, llm: LLMClient, web_search: WebSearchTool,
                 web_visit: WebVisitTool, code_executor: CodeExecutor,
                 experience_manager: Optional[ExperienceManager] = None,
                 max_rounds: int = 15, fetch_top_k: int = 3,
                 explore_max_searches: int = 5):
        self.llm = llm
        self.web_search = web_search
        self.web_visit = web_visit
        self.code_executor = code_executor
        self.experience_manager = experience_manager
        self.max_rounds = max_rounds
        self.explore_max_searches = explore_max_searches
        self.explorer = Explorer(
            skill_manager=_StubSkillManager(),
            web_search=web_search,
            web_visit=web_visit,
            fetch_top_k=fetch_top_k,
        )

    async def solve(self, question: str, context: str = "") -> Dict:
        """Solve a question with auto-injected experience principles."""
        context_section = f"\n## Additional Context:\n{context}" if context else ""

        # Step 1: Retrieve relevant principles and build experience section
        retrieved_principles: List[Tuple[ExperiencePrinciple, float]] = []
        experience_section = ""
        retrieved_principle_ids: List[str] = []

        if self.experience_manager:
            retrieved_principles = await self.experience_manager.retrieve(question)
            if retrieved_principles:
                experience_section = self.experience_manager.build_experience_prompt(
                    retrieved_principles
                )
                retrieved_principle_ids = [p.principle_id for p, _ in retrieved_principles]

        # Step 2: Build prompt with experience injected
        prompt = EVOLVE_MEMORY_PROMPT.format(
            question=question,
            context_section=context_section,
            experience_section=experience_section,
        )

        conversation = [{"role": "user", "content": prompt}]
        actions: List[str] = []
        search_queries: List[str] = []
        search_results: List[Dict] = []
        code_context: List[str] = []
        search_count = 0

        for round_idx in range(self.max_rounds):
            step_response = await self.llm.chat(conversation)
            if not step_response:
                break

            conversation.append({"role": "assistant", "content": step_response})
            step = parse_execution_step(step_response)
            actions.append(step["type"])

            # --- ANSWER ---
            if step["type"] == "answer":
                return {
                    "answer": step["answer"],
                    "actions": actions,
                    "search_queries": search_queries,
                    "search_results": search_results,
                    "num_rounds": round_idx + 1,
                    "conversation_history": conversation,
                    "retrieved_principle_ids": retrieved_principle_ids,
                    "num_principles_retrieved": len(retrieved_principles),
                }

            # --- INVALID ---
            if step["type"] == "invalid":
                err = step.get("parse_error", "")
                conversation.append({
                    "role": "user",
                    "content": (
                        f"Format error: {err}\n"
                        "Please output exactly one action tag: "
                        "<QUERY>, <PAGE_VISIT>, <CODE>, or <ANSWER>."
                    ),
                })
                continue

            # --- QUERY ---
            if step["type"] == "query":
                queries = step.get("queries", [])
                if not queries:
                    conversation.append({
                        "role": "user",
                        "content": 'No queries provided. Use <QUERY>["search keyword"]</QUERY>.',
                    })
                    continue

                if search_count >= self.explore_max_searches:
                    conversation.append({
                        "role": "user",
                        "content": (
                            f"Search limit reached ({self.explore_max_searches} searches).\n"
                            "Use the information gathered so far to answer the question.\n"
                            "Use <ANSWER>...</ANSWER>."
                        ),
                    })
                    continue

                search_data = await self.explorer.execute_search(queries)
                if search_data.get("search_results"):
                    search_count += 1

                search_queries.extend(queries)
                search_results.extend(search_data.get("search_results", []))

                formatted_info = search_data.get("formatted_info", "No results.")
                if len(formatted_info) > 4000:
                    formatted_info = formatted_info[:4000] + "\n...[truncated]"

                conversation.append({
                    "role": "user",
                    "content": (
                        f"Search results for: {'; '.join(queries)}\n\n"
                        f"{formatted_info}\n\n"
                        "IMPORTANT: Search snippets above are brief and may be inaccurate. "
                        "You SHOULD visit the most relevant pages to read their full content before answering.\n"
                        'Next step: Use <PAGE_VISIT>{"urls": ["url1", ...], "goal": "what to extract"}</PAGE_VISIT> '
                        "to read page details, or <ANSWER>...</ANSWER> if you already have enough detail."
                    ),
                })
                continue

            # --- PAGE_VISIT ---
            if step["type"] == "page_visit":
                page_visit_data = step.get("page_visit") or {}
                urls = page_visit_data.get("urls", [])
                goal = page_visit_data.get("goal", "")

                if not urls:
                    conversation.append({
                        "role": "user",
                        "content": (
                            'No URLs provided. Use <PAGE_VISIT>{"urls": ["url1", "url2"], "goal": "what to extract"}</PAGE_VISIT>.'
                        ),
                    })
                    continue

                visit_data = await self.explorer.execute_page_visit(urls, goal=goal)

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

                conversation.append({
                    "role": "user",
                    "content": (
                        f"Page visit results for {len(urls)} URL(s):\n\n"
                        f"{formatted_info}\n\n"
                        "You now have detailed page content. Use it to answer the question.\n"
                        "Use <ANSWER>...</ANSWER> to provide your final answer.\n"
                        "Or <PAGE_VISIT>...</PAGE_VISIT> / <QUERY>...</QUERY> if you need more information."
                    ),
                })
                continue

            # --- CODE ---
            if step["type"] == "run_code":
                code = step.get("run_code", "")
                feedback = self._execute_run_code(code, code_context)
                conversation.append({
                    "role": "user",
                    "content": (
                        f"Code execution feedback:\n{feedback}\n\n"
                        "Continue with your next step. You can:\n"
                        "- <CODE>...</CODE> to run more code\n"
                        "- <QUERY>...</QUERY> to search the web\n"
                        "- <ANSWER>...</ANSWER> to give your final answer"
                    ),
                })
                continue

            # --- Any other type ---
            conversation.append({
                "role": "user",
                "content": (
                    "That action is not available in this mode. "
                    "Available actions: <QUERY>, <PAGE_VISIT>, <CODE>, <ANSWER>."
                ),
            })

        # Max rounds reached
        return {
            "answer": "",
            "actions": actions,
            "search_queries": search_queries,
            "search_results": search_results,
            "num_rounds": self.max_rounds,
            "conversation_history": conversation,
            "retrieved_principle_ids": retrieved_principle_ids,
            "num_principles_retrieved": len(retrieved_principles),
        }

    def _execute_run_code(self, code: str, code_context: List[str]) -> str:
        """Execute Python code with accumulated code_context prepended."""
        if not code:
            return "No code provided. Put your Python code inside <CODE>...</CODE>."

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
                feedback = f"Code executed successfully.\nOutput:\n{stdout}"
            else:
                feedback = (
                    "Code executed successfully, but produced no output.\n"
                    "Make sure to use print() to output the results you want to see."
                )
        else:
            feedback = (
                f"Code execution failed.\n"
                f"stdout: {exec_result['stdout']}\n"
                f"stderr: {exec_result['stderr']}"
            )

        if len(feedback) > 2500:
            feedback = feedback[:2500] + "..."
        return feedback


class _StubSkillManager:
    """Minimal stub so Explorer can be instantiated without a real SkillManager."""
    skills = []
