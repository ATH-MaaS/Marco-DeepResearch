"""Embedding-based skill retrieval using Qwen3-Embedding (or any OpenAI-compatible embedding API).

Provides dense vector retrieval as an alternative to SkillManager's fuzzy matching.
Used in 'full' evaluation mode for cross-question skill reuse.

Dual-channel retrieval:
  - Channel 1: skill description (name + use_when)
  - Channel 2: source_question (the original question that generated the skill)
  - Final score = max(sim_channel1, sim_channel2) per skill

This ensures that skills generated from similar questions are retrieved even
when the use_when description is too abstract to match surface-level query text.

Usage:
    retriever = EmbeddingRetriever.from_config(config)
    await retriever.index_skills(skills_list)
    relevant = await retriever.retrieve(question, top_k=3)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI


class EmbeddingRetriever:
    """Dense vector retrieval for skill library using an embedding model.

    Dual-channel retrieval: indexes both skill description (name + use_when)
    and source_question (the original question that generated the skill).
    Final similarity = max(sim_description, sim_source_question) per skill.
    """

    def __init__(self, api_base: str, api_key: str, model: str = "text-embedding-v4",
                 dimensions: Optional[int] = None, batch_size: int = 32,
                 alpha: Optional[float] = None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=60.0)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        # alpha is the weight for the description channel in
        # score = alpha * desc_sim + (1 - alpha) * question_sim.
        # When alpha is None we fall back to max(desc_sim, question_sim).
        self.alpha = alpha
        self._skills: List[Dict] = []
        self._desc_matrix: Optional[np.ndarray] = None  # (N, D) L2-normalized description embeddings
        self._question_matrix: Optional[np.ndarray] = None  # (N, D) L2-normalized source_question embeddings
        self._has_question: Optional[np.ndarray] = None  # (N,) bool mask: whether skill has source_question

    @classmethod
    def from_config(cls, config) -> Optional["EmbeddingRetriever"]:
        """Create from config['embedding'] section. Returns None if not configured."""
        emb_cfg = config.get("embedding")
        if not emb_cfg:
            return None
        return cls(
            api_base=emb_cfg["api_base"],
            api_key=emb_cfg["api_key"],
            model=emb_cfg.get("model", "text-embedding-v4"),
            dimensions=emb_cfg.get("dimensions"),
            batch_size=emb_cfg.get("batch_size", 32),
            alpha=emb_cfg.get("alpha"),
        )

    # ==================== Embedding API ====================

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts (with retry on rate limit)."""
        import asyncio
        import random
        from openai import RateLimitError

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            kwargs: Dict = {"model": self.model, "input": batch}
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions

            max_retries = 6
            for attempt in range(max_retries + 1):
                try:
                    response = await self.client.embeddings.create(**kwargs)
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                except RateLimitError:
                    if attempt >= max_retries:
                        raise
                    sleep_time = min(2 ** attempt + random.uniform(0, 1), 30)
                    print(f"[EmbeddingRetriever] Rate limit, retry {attempt+1}/{max_retries} in {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
        return all_embeddings

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        result = await self._get_embeddings([text])
        return result[0]

    # ==================== Skill Text ====================

    def _skill_to_text(self, skill: Dict) -> str:
        """Convert a skill dict to embedding text: skill_name + use_when only."""
        parts = []
        if skill.get("skill_name"):
            parts.append(skill["skill_name"])
        if skill.get("use_when"):
            parts.append(skill["use_when"])
        return " | ".join(parts) if parts else ""

    # ==================== Internal Helpers ====================

    def _normalize_vec(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector or matrix along last axis."""
        if vec.ndim == 1:
            norm = np.linalg.norm(vec)
            return vec / max(norm, 1e-12)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vec / norms

    def _compute_scores(self, query_emb: np.ndarray) -> np.ndarray:
        """Compute per-skill scores combining description and source_question channels.

        If self.alpha is set, score = alpha * desc_sim + (1 - alpha) * question_sim
        on skills that have a source_question (others fall back to desc_sim).
        Otherwise, falls back to max(desc_sim, question_sim).
        """
        desc_sims = self._desc_matrix @ query_emb  # (N,)
        if self._question_matrix is not None and np.any(self._has_question):
            question_sims = self._question_matrix @ query_emb  # (N,)
            if self.alpha is None:
                # Default: max-pool (legacy behavior). Suppress question score
                # for skills lacking a source_question.
                q_eff = np.where(self._has_question, question_sims, -1.0)
                return np.maximum(desc_sims, q_eff)
            # Weighted sum (preferred). Fall back to desc_sim where no source_question.
            weighted = self.alpha * desc_sims + (1 - self.alpha) * question_sims
            return np.where(self._has_question, weighted, desc_sims)
        return desc_sims

    # ==================== Index & Retrieve ====================

    async def index_skills(self, skills: List[Dict]):
        """Build dual-channel embedding index for a list of skills."""
        if not skills:
            self._skills = []
            self._desc_matrix = None
            self._question_matrix = None
            self._has_question = None
            return

        indexable = [s for s in skills if not s.get("builtin") and self._skill_to_text(s).strip()]
        if not indexable:
            self._skills = []
            self._desc_matrix = None
            self._question_matrix = None
            self._has_question = None
            return

        self._skills = indexable

        # Channel 1: description embeddings (name + use_when)
        desc_texts = [self._skill_to_text(s) for s in indexable]
        desc_embeddings = await self._get_embeddings(desc_texts)
        self._desc_matrix = self._normalize_vec(
            np.array(desc_embeddings, dtype=np.float32))

        # Channel 2: source_question embeddings (if available)
        has_question = [bool(s.get("source_question", "").strip()) for s in indexable]
        self._has_question = np.array(has_question, dtype=bool)

        if np.any(self._has_question):
            question_texts = [
                s.get("source_question", "").strip() if h else ""
                for s, h in zip(indexable, has_question)
            ]
            # Only embed non-empty questions; use zeros for skills without source_question
            non_empty_indices = [i for i, h in enumerate(has_question) if h]
            non_empty_texts = [question_texts[i] for i in non_empty_indices]
            non_empty_embeddings = await self._get_embeddings(non_empty_texts)

            dim = len(desc_embeddings[0])
            q_matrix = np.zeros((len(indexable), dim), dtype=np.float32)
            for idx, emb in zip(non_empty_indices, non_empty_embeddings):
                q_matrix[idx] = emb
            self._question_matrix = self._normalize_vec(q_matrix)
            # Zero out rows without source_question (already zero, but ensure norm doesn't distort)
            self._question_matrix[~self._has_question] = 0.0

            n_with_q = int(np.sum(self._has_question))
            print(f"[EmbeddingRetriever] Indexed {len(indexable)} skills "
                  f"({n_with_q} with source_question, dual-channel)")
        else:
            self._question_matrix = None
            print(f"[EmbeddingRetriever] Indexed {len(indexable)} skills (description-only)")

    async def retrieve(self, question: str, top_k: int = 3,
                       threshold: float = 0.3) -> List[Dict]:
        """Retrieve top-k most relevant skills via dual-channel cosine similarity."""
        if not self._skills or self._desc_matrix is None:
            return []

        query_emb = np.array(await self._get_embedding(question), dtype=np.float32)
        query_emb = self._normalize_vec(query_emb)
        if np.linalg.norm(query_emb) < 1e-12:
            return []

        sims = self._compute_scores(query_emb)
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if sims[idx] >= threshold:
                results.append(self._skills[idx])

        top_score = float(sims[top_indices[0]]) if len(top_indices) > 0 else 0.0
        if results:
            print(f"[EmbeddingRetriever] Retrieved {len(results)} skills "
                  f"(top score: {top_score:.3f}, threshold: {threshold})")
        else:
            print(f"[EmbeddingRetriever] No skills above threshold "
                  f"(top score: {top_score:.3f}, threshold: {threshold})")
        return results

    async def retrieve_with_scores(self, question: str, top_k: int = 3,
                                   threshold: float = 0.3) -> List[tuple]:
        """Like retrieve(), but returns List[(skill_dict, cosine_score)]."""
        if not self._skills or self._desc_matrix is None:
            return []

        query_emb = np.array(await self._get_embedding(question), dtype=np.float32)
        query_emb = self._normalize_vec(query_emb)
        if np.linalg.norm(query_emb) < 1e-12:
            return []

        sims = self._compute_scores(query_emb)
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = round(float(sims[idx]), 4)
            if sims[idx] >= threshold:
                results.append((self._skills[idx], score))

        top_score = float(sims[top_indices[0]]) if len(top_indices) > 0 else 0.0
        if results:
            print(f"[EmbeddingRetriever] Retrieved {len(results)} skills "
                  f"(top score: {top_score:.3f}, threshold: {threshold})")
        else:
            print(f"[EmbeddingRetriever] No skills above threshold "
                  f"(top score: {top_score:.3f}, threshold: {threshold})")
        return results

    # ==================== Incremental Update ====================

    async def add_skill_auto(self, skill: Dict):
        """Add a new skill to the index, computing its embedding automatically.

        Handles both description and source_question channels.
        """
        if skill.get("builtin"):
            return
        text = self._skill_to_text(skill)
        if not text.strip():
            return

        # Description embedding
        desc_emb = await self._get_embedding(text)
        desc_vec = self._normalize_vec(
            np.array(desc_emb, dtype=np.float32).reshape(1, -1))

        # Source question embedding
        source_q = (skill.get("source_question") or "").strip()
        has_q = bool(source_q)

        self._skills.append(skill)

        # Update description matrix
        if self._desc_matrix is not None:
            self._desc_matrix = np.vstack([self._desc_matrix, desc_vec])
        else:
            self._desc_matrix = desc_vec

        # Update question matrix
        if has_q:
            q_emb = await self._get_embedding(source_q)
            q_vec = self._normalize_vec(
                np.array(q_emb, dtype=np.float32).reshape(1, -1))
        else:
            dim = desc_vec.shape[1]
            q_vec = np.zeros((1, dim), dtype=np.float32)

        if self._question_matrix is not None:
            self._question_matrix = np.vstack([self._question_matrix, q_vec])
        else:
            self._question_matrix = q_vec

        # Update has_question mask
        if self._has_question is not None:
            self._has_question = np.append(self._has_question, has_q)
        else:
            self._has_question = np.array([has_q], dtype=bool)
