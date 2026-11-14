import random
from typing import List, Dict
from datasets import load_dataset
from rapidfuzz import fuzz

from .base import BaseBenchmark


# Difficulty level mapping (higher = harder)
_DIFFICULTY_LEVELS = {
    "Primary School": 1,
    "Junior High School": 2,
    "Senior High School": 3,
    "University": 4,
    "PhD": 5,
}


class WebInstructBenchmark(BaseBenchmark):
    """WebInstruct-verified: open-ended QA dataset covering multi-domain university-level questions.

    Only loads University and PhD difficulty samples by default.
    By default excludes Mathematics category; set exclude_math=False to include it.
    """

    MIN_DIFFICULTY_LEVEL = 4  # University

    def __init__(self, exclude_math: bool = True):
        self.exclude_math = exclude_math

    def load(self, data_dir: str, num_samples: int = -1, subset: str = None, seed: int = None) -> List[Dict]:
        ds = load_dataset("TIGER-Lab/WebInstruct-verified", split="train",
                          cache_dir=data_dir)

        items = []
        for row in ds:
            category = row.get("category", "")
            difficulty = row.get("difficulty", "")

            # Filter by subset (category)
            if subset and str(category).lower() != subset.lower():
                continue

            # Filter by difficulty: keep University and PhD only
            diff_level = _DIFFICULTY_LEVELS.get(difficulty, 0)
            if diff_level < self.MIN_DIFFICULTY_LEVEL:
                continue

            # Exclude Mathematics category by default
            if self.exclude_math and str(category).strip().lower() == "mathematics":
                continue

            items.append({
                "id": str(row.get("id", len(items))),
                "question": row.get("question", ""),
                "answer": str(row.get("answer", "")),
                "options": [],
                "context": "",
                "ground_truth": str(row.get("answer", "")),
                "domain": str(category) if category else "general",
                "metadata": {
                    "answer_type": row.get("answer_type", ""),
                    "difficulty": difficulty,
                },
            })

        if num_samples > 0 and num_samples < len(items):
            rng = random.Random(seed) if seed is not None else random.Random()
            items = rng.sample(items, num_samples)

        return items

    def format_question(self, item: Dict) -> str:
        return item["question"]

    def get_ground_truth(self, item: Dict) -> str:
        return item["ground_truth"]

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> Dict:
        pred = (prediction or "").strip()
        gt = (ground_truth or "").strip()

        if not pred or not gt:
            return {"correct": False, "score": 0.0, "detail": f"empty: pred={pred!r}, gt={gt!r}"}

        # Try numeric comparison for float-like answers
        try:
            pred_num = float(pred.replace(",", "").replace("$", "").strip())
            gt_num = float(gt.replace(",", "").replace("$", "").strip())
            # Allow 1% relative tolerance or absolute tolerance of 0.01
            if gt_num == 0:
                correct = abs(pred_num - gt_num) < 0.01
            else:
                correct = abs(pred_num - gt_num) / abs(gt_num) < 0.01 or abs(pred_num - gt_num) < 0.01
            if correct:
                return {"correct": True, "score": 1.0, "detail": f"numeric_match: pred={pred_num}, gt={gt_num}"}
        except (ValueError, TypeError):
            pass

        # Fallback: fuzzy string matching
        score = fuzz.token_set_ratio(pred.lower(), gt.lower()) / 100.0
        correct = score >= 0.8
        return {
            "correct": correct,
            "score": score,
            "detail": f"fuzzy_score={score:.2f}",
        }
