import json
import os
import re
import uuid
from datetime import datetime
from textwrap import dedent
from typing import List, Dict, Optional
from rapidfuzz import fuzz

from core.skill_schema import (
    normalize_skill,
    skill_code,
    skill_components,
    skill_has_payload,
    skill_use_when,
    skill_workflow,
)


class SkillManager:

    def __init__(self, library_path: str = "skills/skill_library.json",
                 max_skills: int = 200, similarity_threshold: float = 0.7):
        self.library_path = library_path
        self.max_skills = max_skills
        self.similarity_threshold = similarity_threshold
        self.skills: List[Dict] = []
        self.load_library()

    @classmethod
    def from_config(cls, config) -> "SkillManager":
        s = config["skill"]
        return cls(
            library_path=s["library_path"],
            max_skills=s["max_skills"],
            similarity_threshold=s["similarity_threshold"],
        )

    def load_library(self):
        if os.path.exists(self.library_path):
            with open(self.library_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.skills = [
                normalize_skill(s) for s in loaded
                if isinstance(s, dict) and not s.get("builtin")
            ]
        else:
            self.skills = []

    def save_library(self):
        dirpath = os.path.dirname(self.library_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self.skills = [
            normalize_skill(s) for s in self.skills
            if isinstance(s, dict) and not s.get("builtin")
        ]
        with open(self.library_path, "w", encoding="utf-8") as f:
            json.dump(self.skills, f, ensure_ascii=False, indent=2)

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant skills."""
        raw_query_text = (question or "").strip()
        query_text = self._normalize_retrieval_query(raw_query_text)
        threshold_score = self.similarity_threshold * 100
        print(f"[SkillRetrieve] Query: {query_text}")

        results = []
        if self.skills:
            scored = []
            for skill in self.skills:
                normalized = normalize_skill(skill)
                match_text = (
                    f"{normalized.get('skill_name', '')} "
                    f"{skill_use_when(normalized)}"
                ).strip()
                score = fuzz.token_set_ratio(query_text.lower(), match_text.lower())
                scored.append((score, normalized, match_text))
            scored.sort(key=lambda x: x[0], reverse=True)

            results = [s for score, s, _ in scored[:top_k] if score > threshold_score]
        else:
            print("[SkillRetrieve] No custom skills loaded from library.")

        print(f"[SkillRetrieve] Returned skills: {[s.get('skill_name', '') for s in results]}")
        return results

    def get_all_skills(self) -> List[Dict]:
        """Return all current library skills."""
        return list(self.skills)

    def _normalize_retrieval_query(self, question: str, max_chars: int = 600) -> str:
        """Normalize long benchmark prompt into a cleaner retrieval query.

        - Remove MCQ option lines like 'A. ...', 'B. ...'
        - Collapse whitespace
        - Truncate to avoid very long noisy prompts
        """
        if not question:
            return ""

        lines = []
        for line in question.splitlines():
            stripped = line.strip()
            if re.match(r"^[A-J]\.\s+", stripped):
                continue
            lines.append(stripped)

        normalized = " ".join([x for x in lines if x])
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if len(normalized) > max_chars:
            normalized = normalized[:max_chars]
        return normalized

    def add_skill(self, skill: Dict) -> str:
        skill = normalize_skill(skill)
        if not skill_has_payload(skill):
            raise ValueError("Skill must contain at least one non-empty field: workflow or code.")

        # Check if a skill with the same normalized name already exists
        new_name_norm = self._normalize_skill_name(skill.get("skill_name", ""))
        existing_idx = None
        if new_name_norm:
            for i, s in enumerate(self.skills):
                normalized = normalize_skill(s)
                if self._normalize_skill_name(normalized.get("skill_name", "")) == new_name_norm:
                    existing_idx = i
                    break

        if existing_idx is not None:
            # Replace in-place, preserving identity and usage stats
            old = normalize_skill(self.skills[existing_idx])
            skill["skill_id"] = old.get("skill_id", f"skill_{uuid.uuid4().hex[:8]}")
            skill["usage_count"] = old.get("usage_count", 0)
            skill["success_count"] = old.get("success_count", 0)
            skill["version"] = old.get("version", 1) + 1
            skill["created_at"] = old.get("created_at", datetime.now().isoformat())
            skill["evolved_from"] = old.get("skill_id")
            self.skills[existing_idx] = skill
        else:
            skill_id = f"skill_{uuid.uuid4().hex[:8]}"
            skill["skill_id"] = skill_id
            skill.setdefault("usage_count", 0)
            skill.setdefault("success_count", 0)
            skill.setdefault("version", 1)
            skill.setdefault("created_at", datetime.now().isoformat())
            skill.setdefault("evolved_from", None)
            self.skills.append(skill)
            if len(self.skills) > self.max_skills:
                self.skills.sort(key=lambda s: s.get("usage_count", 0))
                self.skills = self.skills[1:]

        self.save_library()
        return skill["skill_id"]

    def get_skill_by_id(self, skill_id: str) -> Optional[Dict]:
        for skill in self.skills:
            normalized = normalize_skill(skill)
            if normalized.get("skill_id") == skill_id:
                return normalized
        return None

    def get_skill_by_name(self, skill_name: str) -> Optional[Dict]:
        target = self._normalize_skill_name(skill_name)
        if not target:
            return None
        for skill in self.skills:
            normalized = normalize_skill(skill)
            if self._normalize_skill_name(normalized.get("skill_name", "")) == target:
                return normalized
        return None

    def evolve_skill(self, skill_id: str, new_content: str = "",
                     reason: str = "", workflow: Optional[str] = None,
                     code: Optional[str] = None,
                     use_when: Optional[str] = None,
                     skill_name: Optional[str] = None,
                     # backward compat aliases
                     knowledge: Optional[str] = None,
                     name: Optional[str] = None) -> bool:
        # Accept old param names as fallback
        if workflow is None and knowledge is not None:
            workflow = knowledge
        for i, skill in enumerate(self.skills):
            if skill["skill_id"] == skill_id:
                normalized = normalize_skill(skill)

                # Backward compatibility: old evolve only passed new_content.
                if workflow is None and code is None and new_content:
                    if skill_code(normalized) and not skill_workflow(normalized):
                        code = new_content
                    else:
                        workflow = new_content

                if workflow is not None:
                    normalized["workflow"] = (workflow or "").strip()
                if code is not None:
                    normalized["code"] = (code or "").strip()
                if use_when is not None:
                    normalized["use_when"] = use_when.strip()
                if skill_name is not None:
                    normalized["skill_name"] = skill_name.strip()

                if not skill_has_payload(normalized):
                    return False

                normalized["version"] = normalized.get("version", 1) + 1
                normalized["evolved_from"] = skill_id
                normalized["evolved_reason"] = reason
                normalized["evolved_at"] = datetime.now().isoformat()
                self.skills[i] = normalized
                self.save_library()
                return True
        return False

    def remove_skill(self, skill_id: str) -> bool:
        for i, skill in enumerate(self.skills):
            if skill.get("skill_id") == skill_id:
                self.skills.pop(i)
                self.save_library()
                return True
        return False

    def update_stats(self, skill_id: str, success: bool):
        for skill in self.skills:
            if skill["skill_id"] == skill_id:
                skill["usage_count"] = skill.get("usage_count", 0) + 1
                if success:
                    skill["success_count"] = skill.get("success_count", 0) + 1
                self.save_library()
                return

    def format_skills_for_prompt(self, skills: List[Dict]) -> str:
        """Format skills list as a concise index for the prompt.

        Only shows metadata (skill_name, use_when, components, call format).
        The actual workflow/code content is revealed on demand when the skill is activated.
        """
        if not skills:
            return "(No skills available)"
        parts = []
        for i, raw_skill in enumerate(skills, 1):
            s = normalize_skill(raw_skill)
            components = skill_components(s)
            header = dedent(
                f"""\
                ### Skill {i}: {s.get('skill_name', '')}
                - use_when: {skill_use_when(s)}
                - Components: {', '.join(components) if components else 'none'}
                """
            ).strip()
            sections = [header]

            skill_call_example = s.get("skill_name", "")
            sections.append(f"- Call format: <ACTIVATE_SKILL>{skill_call_example}</ACTIVATE_SKILL>")

            components = skill_components(s)
            if components:
                effect_parts = []
                if "workflow" in components:
                    effect_parts.append("injects workflow knowledge into context")
                if "code" in components:
                    effect_parts.append("loads Python functions into the execution environment (use <CODE> to call them)")
                sections.append(f"- Effect on call: {'; '.join(effect_parts)}.")

            parts.append("\n".join(sections))
        return "\n\n".join(parts)

    def find_similar(self, description: str, threshold: float = None) -> Optional[Dict]:
        threshold = threshold or self.similarity_threshold
        best_score = 0
        best_skill = None
        for raw_skill in self.skills:
            skill = normalize_skill(raw_skill)
            match_text = (
                f"{skill['skill_name']} {skill_use_when(skill)}"
            )
            score = fuzz.token_set_ratio(description.lower(), match_text.lower()) / 100.0
            if score > best_score:
                best_score = score
                best_skill = skill
        if best_score >= threshold:
            return best_skill
        return None

    def _normalize_skill_name(self, value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text
