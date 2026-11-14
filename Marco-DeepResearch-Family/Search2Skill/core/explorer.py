"""Search utility for the EXPLORE action.

Handles: search -> format results for model review.
Model then selects URLs via PAGE_VISIT for fetch+compress.
No longer manages conversations or multi-round judge loops —
those are now part of the main agent loop.
"""

import json
import re
from typing import Dict, List, Optional, Union

from json_repair import repair_json
from core.skill_manager import SkillManager
from core.skill_schema import (
    normalize_skill,
    skill_code,
    skill_components,
    skill_has_payload,
    skill_use_when,
    skill_workflow,
)
from tools.web_search import (
    WebSearchTool, dedup_results,
)
from tools.web_visit import WebVisitTool


class Explorer:
    """Search utility: search -> format results, page_visit -> fetch+compress.

    Two-phase flow:
    1. execute_search: search queries and return candidate results for the model to review.
    2. execute_page_visit: model selects URLs to visit, fetch+compress pages.

    Used by ExplorationAgent to execute the environment side of EXPLORE actions.
    """

    def __init__(self, skill_manager: SkillManager,
                 web_search: WebSearchTool, web_visit: WebVisitTool,
                 fetch_top_k: int = 3):
        self.skill_manager = skill_manager
        self.web_search = web_search
        self.web_visit = web_visit
        self.fetch_top_k = fetch_top_k

    async def execute_search(self, queries: Union[str, List[str]], knowledge_gaps: str = "") -> Dict:
        """Execute search round: search queries -> return candidate results for model review.

        Args:
            queries: A single query string or a list of query strings.
            knowledge_gaps: Context about what info is needed.

        Returns:
            {
                "formatted_info": str,       # formatted search results for the model
                "search_results": list,       # raw search result dicts (deduplicated)
            }
        """
        if isinstance(queries, str):
            queries = [queries]

        if not knowledge_gaps:
            knowledge_gaps = "; ".join(queries)

        # Batch search all queries
        per_query_results = await self.web_search.batch_search(queries)

        # Merge and deduplicate across all queries
        all_candidates: List[Dict] = []
        for q in queries:
            items = per_query_results.get(q, [])
            all_candidates.extend(items)
        all_candidates = dedup_results(all_candidates, self.web_search.max_results * len(queries))

        if not all_candidates:
            return {
                "formatted_info": "No search results found for these queries.",
                "search_results": [],
            }

        print(f"    [Explore Search] {len(queries)} queries -> {len(all_candidates)} candidates")

        formatted_info = self._format_search_snippets(all_candidates)

        return {
            "formatted_info": formatted_info,
            "search_results": all_candidates,
        }

    async def execute_page_visit(self, urls: Union[str, List[str]], goal: str = "") -> Dict:
        """Fetch and compress pages for the selected URLs.

        Called after the model reviews search results and selects URLs via PAGE_VISIT.

        Args:
            urls: A single URL string or a list of URL strings.
            goal: The information goal (what to extract from pages).

        Returns:
            {
                "formatted_info": str,       # formatted extracted info
                "visited_pages": list,        # raw visit result dicts
            }
        """
        if isinstance(urls, str):
            urls = [urls]

        if not goal:
            goal = "Extract relevant information from the page."

        visit_results = await self.web_visit.visit(urls, goal)

        for vr in visit_results:
            status = f"found={vr['found']}" if vr.get("extracted_content") else "no content"
            print(
                f"      - {vr['url'][:80]} [{vr['fetch_source']}] "
                f"({vr['content_length']} chars -> {status})"
            )

        formatted_info = self._format_visit_results(visit_results, [])

        return {
            "formatted_info": formatted_info,
            "visited_pages": visit_results,
        }

    # ==================== Skill Parsing (for GENERATE_SKILL) ====================

    def parse_and_register_skill(
        self,
        skill_data: Dict,
        current_skills: Optional[List[Dict]] = None,
        preferred_skill_names: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """Normalize and register a skill from GENERATE_SKILL action.

        Args:
            skill_data: Raw skill dict from parse_execution_step (skill_data field).

        Returns:
            Normalized skill dict with skill_id, or None if invalid.
        """
        normalized = self._normalize_one_skill(skill_data)
        if not normalized:
            return None

        skill_to_save = normalized["skill"]
        existing_skill = self._resolve_existing_skill(
            skill_to_save,
            current_skills=current_skills or [],
            preferred_skill_names=preferred_skill_names or [],
        )
        if existing_skill:
            updated_skill = self._evolve_existing_skill(existing_skill, skill_to_save)
            if not updated_skill:
                return None
            comps = ", ".join(skill_components(updated_skill)) or "none"
            print(
                f"    [Generate Skill] Evolved: {updated_skill.get('skill_name', '?')} "
                f"(components={comps})"
            )
            return updated_skill

        skill_id = self.skill_manager.add_skill(skill_to_save)
        skill_to_save["skill_id"] = skill_id
        skill_to_save["action"] = "add"

        comps = ", ".join(skill_components(skill_to_save)) or "none"
        print(f"    [Generate Skill] Registered: {skill_to_save.get('skill_name', '?')} (components={comps})")

        return skill_to_save

    # ==================== Formatting Helpers ====================

    def _format_visit_results(self, visit_results: List[Dict],
                               candidates: List[Dict]) -> str:
        """Format visit results for the model, including supplementary knowledge."""
        def _stringify(value) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return str(value).strip()

        parts = []
        snippet_map = {c.get("url", ""): c.get("content", "") for c in candidates}

        for i, vr in enumerate(visit_results, 1):
            url = vr.get("url", "")
            if vr.get("found") and vr.get("extracted_content"):
                section = f"[{i}] {url}\n{vr['extracted_content']}"

                supplementary = _stringify(vr.get("supplementary_knowledge", ""))
                if supplementary:
                    section += f"\n[Supplementary knowledge] {supplementary}"

                parts.append(section)
            elif vr.get("extracted_content"):
                parts.append(
                    f"[{i}] {url}\n"
                    f"[No relevant information found on this page]"
                )
            else:
                snippet = snippet_map.get(url, "")
                if snippet:
                    parts.append(f"[{i}] {url}\n[Snippet only, fetch failed]\n{snippet}")
                else:
                    parts.append(f"[{i}] {url}\n[Failed to fetch page content]")

        return "\n\n---\n\n".join(parts)

    def _format_search_snippets(self, results: List[Dict]) -> str:
        """Format search results as snippets (fallback when no URLs selected)."""
        parts = []
        for i, r in enumerate(results, 1):
            content = r.get("content", "")
            url = r.get("url", "")
            title = r.get("title", "")
            parts.append(f"[{i}] {title}\nURL: {url}\n{content}")
        return "\n\n".join(parts)

    # ==================== Skill Normalization ====================

    def _normalize_one_skill(self, entry: Dict) -> Optional[Dict]:
        """Normalize a single raw skill entry into SkillManager format."""
        if not isinstance(entry, dict):
            return None

        skill_name = str(entry.get("skill_name") or entry.get("name") or "").strip()
        if not skill_name:
            return None

        # Accept both new and legacy field names
        workflow = entry.get("workflow", "") or entry.get("knowledge", "")
        code = entry.get("code", "")
        use_when = entry.get("use_when", "") or entry.get("description", "")

        if not isinstance(workflow, str):
            workflow = json.dumps(workflow, ensure_ascii=False)
        if not isinstance(code, str):
            code = json.dumps(code, ensure_ascii=False)
        if not isinstance(use_when, str):
            use_when = str(use_when)
        workflow = workflow.strip()
        code = code.strip()
        use_when = use_when.strip()

        # Auto-generate use_when if not provided
        if not use_when:
            name_readable = " ".join(skill_name.replace("_", " ").split()).lower()
            if workflow and code:
                use_when = f"Reusable skill (workflow + code) for {name_readable}."
            elif code:
                use_when = f"Reusable code function for {name_readable}."
            else:
                use_when = f"Reusable workflow for {name_readable}."

        normalized_skill = normalize_skill({
            "skill_name": skill_name,
            "use_when": use_when,
            "workflow": workflow,
            "code": code,
        })
        if not skill_has_payload(normalized_skill):
            return None

        return {
            "skill": normalized_skill,
        }

    def _resolve_existing_skill(
        self,
        new_skill: Dict,
        current_skills: List[Dict],
        preferred_skill_names: List[str],
    ) -> Optional[Dict]:
        target_name = self._normalize_skill_name(new_skill.get("skill_name", ""))
        if not target_name:
            return None

        current_normalized = []
        for skill in current_skills:
            if not isinstance(skill, dict):
                continue
            normalized = normalize_skill(skill)
            if normalized.get("builtin"):
                continue
            current_normalized.append(normalized)

        for skill in current_normalized:
            if self._normalize_skill_name(skill.get("skill_name", "")) == target_name:
                return skill

        exact_library = self.skill_manager.get_skill_by_name(target_name)
        if exact_library:
            return exact_library

        return None

    def _evolve_existing_skill(self, existing_skill: Dict, new_skill: Dict) -> Optional[Dict]:
        existing_skill = normalize_skill(existing_skill)
        new_skill = normalize_skill(new_skill)

        reason = f"Evolved from exploration for skill_name={existing_skill.get('skill_name', '')}"
        skill_id = existing_skill.get("skill_id", "")
        if skill_id and self.skill_manager.evolve_skill(
            skill_id,
            reason=reason,
            workflow=skill_workflow(new_skill),
            code=skill_code(new_skill),
            use_when=skill_use_when(new_skill) or skill_use_when(existing_skill),
            skill_name=new_skill.get("skill_name") or existing_skill.get("skill_name"),
        ):
            updated = self.skill_manager.get_skill_by_id(skill_id)
            if updated:
                updated["action"] = "evolve"
                return updated

        merged = normalize_skill({
            **existing_skill,
            "use_when": skill_use_when(new_skill) or skill_use_when(existing_skill),
            "skill_name": existing_skill.get("skill_name") or new_skill.get("skill_name"),
            "workflow": skill_workflow(new_skill) or skill_workflow(existing_skill),
            "code": skill_code(new_skill) or skill_code(existing_skill),
        })
        merged["action"] = "evolve"
        merged["evolved_from"] = existing_skill.get("skill_id")
        merged["version"] = int(existing_skill.get("version", 1) or 1) + 1
        return merged

    def _normalize_skill_name(self, value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text

    def _parse_json(self, text: str) -> Optional[Dict]:
        match = re.search(r"```json?\n(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(repair_json(match.group(0)))
            except json.JSONDecodeError:
                pass
        return None
