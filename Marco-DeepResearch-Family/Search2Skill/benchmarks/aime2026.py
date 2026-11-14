import random
from typing import List, Dict
from datasets import load_dataset
from .base import BaseBenchmark
from evaluation.cascade_judge import cascade_answer_judge


class AIME2026Benchmark(BaseBenchmark):
    """AIME 2026: American Invitational Mathematics Examination 2026.

    Dataset: math-ai/aime26, split="test" (30 questions).
    Answers are integers (0-999).
    """

    def load(self, data_dir: str, num_samples: int = -1,
             subset: str = None, seed: int = None) -> List[Dict]:
        ds = load_dataset("math-ai/aime26", split="test",
                          cache_dir=data_dir)

        items = []
        for row in ds:
            question_text = row.get("problem", "")
            answer = str(row.get("answer", ""))

            items.append({
                "id": str(row.get("id", len(items))),
                "question": question_text,
                "options": [],
                "context": "",
                "ground_truth": answer,
                "domain": "mathematics",
                "metadata": {
                    "subfield": "competition_math",
                    "difficulty": "hard",
                },
            })

        if subset:
            items = [it for it in items if it.get("domain", "").lower() == subset.lower()]

        if num_samples > 0 and num_samples < len(items):
            rng = random.Random(seed) if seed is not None else random.Random()
            items = rng.sample(items, num_samples)

        return items

    def format_question(self, item: Dict) -> str:
        return item["question"]

    def get_ground_truth(self, item: Dict) -> str:
        return item["ground_truth"]

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> Dict:
        pred = prediction.strip()
        gt = ground_truth.strip()

        # AIME answers are integers 0-999, try exact numeric match first
        try:
            gt_num = int(gt)
            pred_clean = pred.rstrip(".").strip()
            pred_num = int(pred_clean)
            correct = pred_num == gt_num
            return {"correct": correct, "score": 1.0 if correct else 0.0,
                    "detail": f"numeric_match: pred={pred_num}, gt={gt_num}"}
        except (ValueError, TypeError):
            pass

        # Fall through to cascade judge for non-numeric predictions
        score, detail = cascade_answer_judge(
            ground_truth=gt,
            prediction=pred,
            question=kwargs.get("question", ""),
        )
        return {"correct": score >= 0.5, "score": score,
                "detail": f"cascade_judge: {detail}"}
