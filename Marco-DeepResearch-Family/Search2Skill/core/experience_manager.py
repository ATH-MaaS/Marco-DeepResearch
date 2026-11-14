"""EvolveR-style experience manager: principle accumulation, retrieval, and evolution.

Closely follows EvolveR (https://arxiv.org/abs/2504.xxxxx) design:
  - Principles are one-sentence rules (guiding or cautionary) distilled from trajectories
  - Retrieval via embedding cosine similarity (reuses existing EmbeddingRetriever infra)
  - Metric score = Laplace-smoothed success rate: (success + 1) / (usage + 2)
  - Low-metric principles are periodically pruned

Key differences from EvolveR:
  - No separate VDB server; principles stored in-memory + JSONL export
  - Distillation uses an external LLM client (teacher or actor) rather than actor rollout worker
  - Retrieval is auto-injected into prompt, not triggered by model action
"""

import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think>")


# ---------------------------------------------------------------------------
# Prompts (aligned with EvolveR's prompts.py)
# ---------------------------------------------------------------------------

SUMMARIZE_SUCCESSFUL_TRAJECTORY_PROMPT = """\
You are an expert in analyzing interaction logs to distill generalizable wisdom.
Analyze the following successful interaction trajectory. Your goal is to extract a "Guiding Principle" from it.

A "Guiding Principle" is a concise, one-sentence description of the key strategy or insight that led to success.

[Question]:
{question}

[Trajectory Log]:
{trajectory_log}

Final Outcome: SUCCESS

**Your Task:**
Write exactly ONE sentence that captures the guiding principle. It should be general enough to apply to similar future questions, not specific to this particular question.

[Output]:
"""

SUMMARIZE_FAILED_TRAJECTORY_PROMPT = """\
You are an expert in analyzing interaction logs to find the root cause of failures.
Analyze the following failed interaction trajectory. Your goal is to extract a "Cautionary Principle" from it.

A "Cautionary Principle" is a concise, one-sentence description of the key mistake to avoid and under what circumstances.

[Question]:
{question}

[Trajectory Log]:
{trajectory_log}

Final Outcome: FAILURE

**Your Task:**
Write exactly ONE sentence that captures the cautionary principle. It should be general enough to apply to similar future questions, not specific to this particular question.

[Output]:
"""

MATCH_PRINCIPLE_PROMPT = """\
You are a semantic analysis expert. Determine if two principles describe the same core idea, even if they use different words.

Principle A: "{principle_a}"
Principle B: "{principle_b}"

Do Principle A and Principle B describe the same essential advice or warning?
Please answer with only "Yes" or "No"."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperiencePrinciple:
    principle_id: str
    type: str  # "guiding" or "cautionary"
    description: str
    metric_score: float = 0.5
    usage_count: int = 0
    success_count: int = 0
    source_question: str = ""
    successful_trajectory_ids: List[str] = field(default_factory=list)
    failed_trajectory_ids: List[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "principle_id": self.principle_id,
            "type": self.type,
            "description": self.description,
            "metric_score": self.metric_score,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "source_question": self.source_question,
            "successful_trajectory_ids": self.successful_trajectory_ids,
            "failed_trajectory_ids": self.failed_trajectory_ids,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperiencePrinciple":
        return cls(
            principle_id=d["principle_id"],
            type=d.get("type", "guiding"),
            description=d["description"],
            metric_score=d.get("metric_score", 0.5),
            usage_count=d.get("usage_count", 0),
            success_count=d.get("success_count", 0),
            source_question=d.get("source_question", ""),
            successful_trajectory_ids=d.get("successful_trajectory_ids", []),
            failed_trajectory_ids=d.get("failed_trajectory_ids", []),
            raw_response=d.get("raw_response", ""),
        )


@dataclass
class Trajectory:
    trajectory_id: str
    question: str
    conversation_log: str
    final_outcome: str  # "success" or "failure"
    retrieved_principles: List[str] = field(default_factory=list)
    ground_truth: str = ""


# ---------------------------------------------------------------------------
# ExperienceManager
# ---------------------------------------------------------------------------

class ExperienceManager:
    """Manages experience principles with EvolveR-style lifecycle."""

    def __init__(
        self,
        distill_llm=None,
        embedding_retriever=None,
        retrieve_top_k: int = 3,
        similarity_threshold: float = 0.3,
        clean_low_metric_threshold: float = 0.3,
        max_principles: int = 200,
        export_path: Optional[str] = None,
        max_success_traj_per_principle: int = 3,
        max_failed_traj_per_principle: int = 2,
        max_traj_inject_chars: int = 800,
    ):
        self.principles: List[ExperiencePrinciple] = []
        self.trajectory_buffer: List[Trajectory] = []
        self.trajectory_store: Dict[str, Trajectory] = {}
        self.distill_llm = distill_llm
        self.embedding_retriever = embedding_retriever
        self.retrieve_top_k = retrieve_top_k
        self.similarity_threshold = similarity_threshold
        self.clean_low_metric_threshold = clean_low_metric_threshold
        self.max_principles = max_principles
        self.export_path = export_path
        self.max_success_traj_per_principle = max_success_traj_per_principle
        self.max_failed_traj_per_principle = max_failed_traj_per_principle
        self.max_traj_inject_chars = max_traj_inject_chars

        # Embedding index state (numpy matrices for cosine similarity)
        self._desc_matrix: Optional[np.ndarray] = None
        self._indexed_count: int = 0

        # Stats
        self.total_distilled = 0
        self.total_merged = 0
        self.total_pruned = 0

    # ---- Retrieval ----

    async def retrieve(self, question: str) -> List[Tuple[ExperiencePrinciple, float]]:
        """Retrieve top-k relevant principles for a question via embedding similarity."""
        if not self.principles:
            return []

        if self.embedding_retriever is None:
            return []

        # Rebuild index if new principles were added since last index
        if self._indexed_count != len(self.principles):
            await self._rebuild_index()

        try:
            query_emb = await self.embedding_retriever._get_embedding(question)
            query_vec = self.embedding_retriever._normalize_vec(
                np.array(query_emb, dtype=np.float32)
            )
        except Exception as e:
            print(f"[ExperienceManager] Embedding failed: {e}")
            return []

        if self._desc_matrix is None or len(self._desc_matrix) == 0:
            return []

        scores = self._desc_matrix @ query_vec
        top_indices = np.argsort(scores)[::-1][: self.retrieve_top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= self.similarity_threshold:
                results.append((self.principles[idx], score))
        return results

    async def _rebuild_index(self):
        """Build/rebuild the embedding index for all principles."""
        if not self.principles or self.embedding_retriever is None:
            self._desc_matrix = None
            self._indexed_count = 0
            return

        texts = [p.description for p in self.principles]
        try:
            embeddings = await self.embedding_retriever._get_embeddings(texts)
            matrix = np.array(embeddings, dtype=np.float32)
            self._desc_matrix = self.embedding_retriever._normalize_vec(matrix)
            self._indexed_count = len(self.principles)
        except Exception as e:
            print(f"[ExperienceManager] Index rebuild failed: {e}")
            self._desc_matrix = None
            self._indexed_count = 0

    # ---- Prompt construction ----

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>...</think> blocks and orphan tags."""
        if not text:
            return ""
        stripped = _THINK_RE.sub("", text)
        stripped = _THINK_TAG_RE.sub("", stripped).strip()
        return stripped

    def _format_trajectory_example(self, traj: Trajectory, label: str) -> str:
        log = self._strip_think(traj.conversation_log)
        if len(log) > self.max_traj_inject_chars:
            log = log[: self.max_traj_inject_chars] + "...[truncated]"
        return f"--- Example of {label} ---\n{log}\n"

    def build_experience_prompt(
        self, retrieved: List[Tuple[ExperiencePrinciple, float]]
    ) -> str:
        """Build experience text to inject into the user prompt."""
        if not retrieved:
            return ""

        lines = []
        for i, (principle, score) in enumerate(retrieved, 1):
            lines.append(
                f"[Principle {i}], type: {principle.type}, "
                f"metric score: {principle.metric_score:.2f}, "
                f"description: {principle.description}"
            )
            pos_ids = principle.successful_trajectory_ids[-1:]
            neg_ids = principle.failed_trajectory_ids[-1:]
            for tid in pos_ids:
                traj = self.trajectory_store.get(tid)
                if traj:
                    lines.append(self._format_trajectory_example(traj, "Success"))
            for tid in neg_ids:
                traj = self.trajectory_store.get(tid)
                if traj:
                    lines.append(self._format_trajectory_example(traj, "Failure"))

        return (
            "\n\nRelevant past experience principles:\n"
            + "\n".join(lines)
            + "\n\nUse these principles if helpful. Guiding principles suggest good strategies; "
            "cautionary principles warn against common mistakes.\n"
        )

    # ---- Trajectory collection ----

    def add_trajectory(
        self,
        question: str,
        conversation: List[Dict],
        success: bool,
        retrieved_principle_ids: Optional[List[str]] = None,
        ground_truth: str = "",
    ):
        """Add a completed trajectory to the buffer for later distillation."""
        log_parts = []
        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            content = self._strip_think(content)
            if role == "assistant":
                if len(content) > 500:
                    content = content[:500] + "...[truncated]"
                log_parts.append(f"Assistant: {content}")
            elif role == "user" and len(log_parts) > 0:
                if len(content) > 300:
                    content = content[:300] + "...[truncated]"
                log_parts.append(f"User feedback: {content}")

        traj = Trajectory(
            trajectory_id=uuid4().hex[:12],
            question=question,
            conversation_log="\n".join(log_parts[-10:]),
            final_outcome="success" if success else "failure",
            retrieved_principles=retrieved_principle_ids or [],
            ground_truth=ground_truth,
        )
        self.trajectory_buffer.append(traj)
        self.trajectory_store[traj.trajectory_id] = traj

        if retrieved_principle_ids:
            id_set = set(retrieved_principle_ids)
            for p in self.principles:
                if p.principle_id in id_set:
                    if success:
                        p.successful_trajectory_ids.append(traj.trajectory_id)
                        p.successful_trajectory_ids = p.successful_trajectory_ids[
                            -self.max_success_traj_per_principle :
                        ]
                    else:
                        p.failed_trajectory_ids.append(traj.trajectory_id)
                        p.failed_trajectory_ids = p.failed_trajectory_ids[
                            -self.max_failed_traj_per_principle :
                        ]

    # ---- Metric update ----

    def update_metrics(self, principle_ids: List[str], success: bool):
        """Update usage/success counts for retrieved principles based on outcome."""
        id_set = set(principle_ids)
        for p in self.principles:
            if p.principle_id in id_set:
                p.usage_count += 1
                if success:
                    p.success_count += 1
                p.metric_score = (p.success_count + 1) / (p.usage_count + 2)

    # ---- Reflection (distillation) ----

    async def reflection(self, trajectory_choice_ratio: float = 0.5):
        """Distill new principles from buffered trajectories.

        This is the core EvolveR reflection step:
        1. Select trajectories from buffer
        2. Summarize each into a potential principle via LLM
        3. Deduplicate against existing principles
        4. Merge or create new
        """
        if not self.trajectory_buffer or self.distill_llm is None:
            return

        # Select trajectories (balanced: pick both success and failure)
        buffer = list(self.trajectory_buffer)
        self.trajectory_buffer.clear()

        n_select = max(1, int(len(buffer) * trajectory_choice_ratio))
        successes = [t for t in buffer if t.final_outcome == "success"]
        failures = [t for t in buffer if t.final_outcome == "failure"]

        import random
        selected = []
        # Balanced selection: alternate success/failure
        n_success = min(len(successes), (n_select + 1) // 2)
        n_failure = min(len(failures), n_select // 2)
        if n_success < (n_select + 1) // 2:
            n_failure = min(len(failures), n_select - n_success)
        if n_failure < n_select // 2:
            n_success = min(len(successes), n_select - n_failure)
        selected.extend(random.sample(successes, n_success) if n_success <= len(successes) else successes)
        selected.extend(random.sample(failures, n_failure) if n_failure <= len(failures) else failures)

        print(f"[ExperienceManager] Reflection: {len(selected)} trajectories "
              f"({n_success} success, {n_failure} failure) from buffer of {len(buffer)}")

        new_principles = []
        for traj in selected:
            principle = await self._distill_one(traj)
            if principle:
                new_principles.append(principle)

        # Deduplicate and merge
        for new_p in new_principles:
            merged = await self._try_merge(new_p)
            if merged:
                self.total_merged += 1
            else:
                if len(self.principles) < self.max_principles:
                    self.principles.append(new_p)
                    self.total_distilled += 1
                    self._indexed_count = 0  # force re-index

        if new_principles:
            print(f"[ExperienceManager] Distilled {len(new_principles)} principles, "
                  f"merged {self.total_merged}, total in pool: {len(self.principles)}")

    async def _distill_one(self, traj: Trajectory) -> Optional[ExperiencePrinciple]:
        """Distill a single trajectory into a principle using LLM."""
        if traj.final_outcome == "success":
            prompt = SUMMARIZE_SUCCESSFUL_TRAJECTORY_PROMPT.format(
                question=traj.question[:500],
                trajectory_log=traj.conversation_log[:2000],
            )
        else:
            prompt = SUMMARIZE_FAILED_TRAJECTORY_PROMPT.format(
                question=traj.question[:500],
                trajectory_log=traj.conversation_log[:2000],
            )

        try:
            response = await self.distill_llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            if not response or len(response.strip()) < 10:
                return None
        except Exception as e:
            print(f"[ExperienceManager] Distillation LLM call failed: {e}")
            return None

        raw_response = response or ""
        description = self._strip_think(response).strip()
        if description.startswith("[Output]"):
            description = description.split("[Output]", 1)[-1].strip().lstrip(':').strip()
        description = re.sub(r'```.*?\n|```', '', description, flags=re.DOTALL).strip()
        if len(description) > 3000:
            description = description[:3000]

        # Fallback: if strip_think left nothing useful, retry with thinking disabled
        if len(set(description)) < 5 or len(description) < 10:
            try:
                saved_thinking = getattr(self.distill_llm, 'enable_thinking', False)
                self.distill_llm.enable_thinking = False
                response_retry = await self.distill_llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                self.distill_llm.enable_thinking = saved_thinking
                if response_retry and len(response_retry.strip()) >= 10:
                    raw_response = raw_response + "\n[FALLBACK_NO_THINK]\n" + response_retry
                    description = response_retry.strip()
                    if description.startswith("[Output]"):
                        description = description.split("[Output]", 1)[-1].strip().lstrip(':').strip()
                    description = re.sub(r'```.*?\n|```', '', description, flags=re.DOTALL).strip()
                    if len(description) > 3000:
                        description = description[:3000]
            except Exception:
                pass

        if len(set(description)) < 5 or len(description) < 10:
            return None

        return ExperiencePrinciple(
            principle_id=uuid4().hex[:12],
            type="guiding" if traj.final_outcome == "success" else "cautionary",
            description=description,
            metric_score=0.5,
            usage_count=0,
            success_count=0,
            source_question=traj.question[:200],
            successful_trajectory_ids=[traj.trajectory_id] if traj.final_outcome == "success" else [],
            failed_trajectory_ids=[traj.trajectory_id] if traj.final_outcome == "failure" else [],
            raw_response=raw_response,
        )

    def _merge_trajectory_ids(self, target: ExperiencePrinciple, source: ExperiencePrinciple):
        """Merge trajectory IDs from source into target, keeping recent ones."""
        target.successful_trajectory_ids = (
            target.successful_trajectory_ids + source.successful_trajectory_ids
        )[-self.max_success_traj_per_principle :]
        target.failed_trajectory_ids = (
            target.failed_trajectory_ids + source.failed_trajectory_ids
        )[-self.max_failed_traj_per_principle :]

    async def _try_merge(self, new_principle: ExperiencePrinciple) -> bool:
        """Try to merge a new principle into an existing one.

        Aligned with EvolveR: embedding similarity >= 0.85 enters LLM confirmation,
        < 0.85 is considered different and creates a new principle.
        Returns True if merged, False if should create new.
        """
        if not self.principles:
            return False

        best_idx = 0
        best_score = 0.0

        if self.embedding_retriever is not None:
            if self._indexed_count != len(self.principles):
                await self._rebuild_index()
            if self._desc_matrix is not None:
                try:
                    emb = await self.embedding_retriever._get_embedding(new_principle.description)
                    vec = self.embedding_retriever._normalize_vec(
                        np.array(emb, dtype=np.float32)
                    )
                    scores = self._desc_matrix @ vec
                    best_idx = int(np.argmax(scores))
                    best_score = float(scores[best_idx])

                    if best_score < 0.85:
                        return False
                except Exception:
                    return False

        # Similarity >= 0.85 — use LLM to confirm (aligned with EvolveR)
        candidate = self.principles[best_idx]
        if self.distill_llm is not None:
            try:
                prompt = MATCH_PRINCIPLE_PROMPT.format(
                    principle_a=new_principle.description,
                    principle_b=candidate.description,
                )
                response = await self.distill_llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                response = self._strip_think(response) if response else ""
                if not response or not response.strip().lower().startswith("yes"):
                    return False
            except Exception:
                return False
        else:
            return False

        candidate.usage_count += new_principle.usage_count
        candidate.success_count += new_principle.success_count
        candidate.metric_score = (
            (candidate.success_count + 1) / (candidate.usage_count + 2)
        )
        self._merge_trajectory_ids(candidate, new_principle)
        return True

    # ---- Pruning ----

    def clean_low_metric_principles(self):
        """Remove principles with metric score below threshold."""
        before = len(self.principles)
        self.principles = [
            p for p in self.principles
            if p.usage_count < 3 or p.metric_score >= self.clean_low_metric_threshold
        ]
        pruned = before - len(self.principles)
        if pruned > 0:
            self.total_pruned += pruned
            self._indexed_count = 0  # force re-index
            print(f"[ExperienceManager] Pruned {pruned} low-metric principles, "
                  f"{len(self.principles)} remaining")

    # ---- Persistence ----

    def save(self, path: Optional[str] = None):
        """Export principles and trajectory store to JSONL."""
        path = path or self.export_path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for p in self.principles:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

        traj_path = path.replace(".jsonl", "_trajectories.jsonl")
        referenced_ids = set()
        for p in self.principles:
            referenced_ids.update(p.successful_trajectory_ids)
            referenced_ids.update(p.failed_trajectory_ids)
        with open(traj_path, "w", encoding="utf-8") as f:
            for tid in referenced_ids:
                traj = self.trajectory_store.get(tid)
                if traj:
                    f.write(json.dumps({
                        "trajectory_id": traj.trajectory_id,
                        "question": traj.question,
                        "conversation_log": traj.conversation_log,
                        "final_outcome": traj.final_outcome,
                        "retrieved_principles": traj.retrieved_principles,
                        "ground_truth": traj.ground_truth,
                    }, ensure_ascii=False) + "\n")

    def load(self, path: Optional[str] = None):
        """Import principles and trajectory store from JSONL."""
        path = path or self.export_path
        if not path or not os.path.exists(path):
            return
        self.principles.clear()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self.principles.append(ExperiencePrinciple.from_dict(d))
                except (json.JSONDecodeError, KeyError):
                    continue
        self._indexed_count = 0

        traj_path = path.replace(".jsonl", "_trajectories.jsonl")
        if os.path.exists(traj_path):
            with open(traj_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        traj = Trajectory(
                            trajectory_id=d["trajectory_id"],
                            question=d.get("question", ""),
                            conversation_log=d.get("conversation_log", ""),
                            final_outcome=d.get("final_outcome", ""),
                            retrieved_principles=d.get("retrieved_principles", []),
                            ground_truth=d.get("ground_truth", ""),
                        )
                        self.trajectory_store[traj.trajectory_id] = traj
                    except (json.JSONDecodeError, KeyError):
                        continue

        print(f"[ExperienceManager] Loaded {len(self.principles)} principles, "
              f"{len(self.trajectory_store)} trajectories from {path}")

    def get_stats(self) -> Dict:
        """Return summary stats."""
        return {
            "total_principles": len(self.principles),
            "total_distilled": self.total_distilled,
            "total_merged": self.total_merged,
            "total_pruned": self.total_pruned,
            "avg_metric_score": (
                sum(p.metric_score for p in self.principles) / len(self.principles)
                if self.principles else 0.0
            ),
            "guiding_count": sum(1 for p in self.principles if p.type == "guiding"),
            "cautionary_count": sum(1 for p in self.principles if p.type == "cautionary"),
        }
