import json
import random
from typing import List, Dict
from .base import BaseBenchmark


class MathNumberTheoryBenchmark(BaseBenchmark):
    """MATH Number Theory (Level 3-5): competition-level number theory QA.

    Source: EleutherAI/hendrycks_math, config 'number_theory'.
    Answers extracted from \\boxed{} in solutions.
    Evaluation uses cascade_answer_judge.
    """

    def load(self, data_dir: str, num_samples: int = -1,
             subset: str = None, seed: int = None) -> List[Dict]:
        import os
        if data_dir.endswith(".json"):
            file_path = data_dir
        elif data_dir.endswith(".jsonl"):
            file_path = data_dir
        else:
            file_path = os.path.join(data_dir, "math_nt_test.json")

        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        if subset:
            items = [it for it in items
                     if it.get("metadata", {}).get("difficulty", "").lower() == subset.lower()]

        if num_samples > 0 and num_samples < len(items):
            rng = random.Random(seed) if seed is not None else random.Random()
            items = rng.sample(items, num_samples)

        return items

    def format_question(self, item: Dict) -> str:
        return item["question"]

    def get_ground_truth(self, item: Dict) -> str:
        return item["ground_truth"]

    def evaluate(self, prediction: str, ground_truth: str,
                 question: str = "", **kwargs) -> Dict:
        from evaluation.cascade_judge import cascade_answer_judge

        score, detail = cascade_answer_judge(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
            **kwargs,
        )
        return {
            "correct": score >= 0.5,
            "score": score,
            "detail": detail,
        }
