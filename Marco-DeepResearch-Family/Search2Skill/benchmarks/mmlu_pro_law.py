import json
import random
from typing import List, Dict
from .base import BaseBenchmark
from evaluation.cascade_judge import cascade_answer_judge


class MMLUProSubsetBenchmark(BaseBenchmark):
    """MMLU-Pro subset: 10-option MCQ with cascade_answer_judge fallback.

    Works for any MMLU-Pro subset (law, engineering, etc.) stored as JSON
    with fields: id, question, options, ground_truth, domain, metadata.
    """

    OPTION_LETTERS = "ABCDEFGHIJ"

    def load(self, data_dir: str, num_samples: int = -1,
             subset: str = None, seed: int = None) -> List[Dict]:
        import os
        if data_dir.endswith(".json"):
            file_path = data_dir
        else:
            file_path = os.path.join(data_dir, "test.json")

        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        if subset:
            items = [it for it in items
                     if it.get("domain", "").lower() == subset.lower()]

        if num_samples > 0 and num_samples < len(items):
            rng = random.Random(seed) if seed is not None else random.Random()
            items = rng.sample(items, num_samples)

        return items

    def format_question(self, item: Dict) -> str:
        q = item["question"]
        if item["options"]:
            q += "\n\n" + "\n".join(item["options"])
        return q

    def get_ground_truth(self, item: Dict) -> str:
        return item["ground_truth"]

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> Dict:
        pred = prediction.strip()
        gt = ground_truth.strip().upper()

        if len(gt) == 1 and gt in self.OPTION_LETTERS:
            if len(pred) <= 2 and pred and pred[0].upper() in self.OPTION_LETTERS:
                pred_letter = pred[0].upper()
                correct = pred_letter == gt
                return {"correct": correct, "score": 1.0 if correct else 0.0,
                        "detail": f"pred={pred_letter}, gt={gt}"}

            question = kwargs.get("question", "")
            score, detail = cascade_answer_judge(
                ground_truth=gt, prediction=pred, question=question)
            return {"correct": score >= 0.5, "score": score,
                    "detail": f"cascade_judge: {detail}"}

        score, detail = cascade_answer_judge(
            ground_truth=gt, prediction=pred,
            question=kwargs.get("question", ""))
        return {"correct": score >= 0.5, "score": score,
                "detail": f"cascade_judge: {detail}"}


MMLUProLawBenchmark = MMLUProSubsetBenchmark
