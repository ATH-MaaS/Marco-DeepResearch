import random
from typing import List, Dict
from datasets import load_dataset
from rapidfuzz import fuzz
from .base import BaseBenchmark
from evaluation.cascade_judge import cascade_answer_judge


class SuperGPQABenchmark(BaseBenchmark):
    """SuperGPQA: graduate-level expert knowledge QA."""

    OPTION_LETTERS = "ABCDEFGHIJ"

    def load(self, data_dir: str, num_samples: int = -1, subset: str = None, seed: int = None) -> List[Dict]:
        ds = load_dataset("m-a-p/SuperGPQA", split="train",
                          cache_dir=data_dir)

        items = []
        for row in ds:
            if subset and row.get("discipline", "").lower() != subset.lower():
                continue

            options = row.get("options", [])
            if options and isinstance(options, list):
                formatted_options = [
                    f"{self.OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options) if opt
                ]
            else:
                formatted_options = []

            items.append({
                "id": str(len(items)),
                "question": row.get("question", ""),
                "options": formatted_options,
                "context": "",
                "ground_truth": row.get("answer_letter", ""),
                "domain": row.get("discipline", "general"),
                "metadata": {
                    "subfield": row.get("subfield", ""),
                    "difficulty": row.get("difficulty", ""),
                },
            })

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
        gt = ground_truth.strip()

        # Multiple-choice (single-letter answer)
        if len(gt) == 1 and gt.upper() in self.OPTION_LETTERS:
            # Model output is a single letter, compare directly
            if len(pred) <= 2 and pred and pred[0].upper() in self.OPTION_LETTERS:
                pred_letter = pred[0].upper()
                correct = pred_letter == gt.upper()
                return {"correct": correct, "score": 1.0 if correct else 0.0,
                        "detail": f"pred={pred_letter}, gt={gt}"}

            # Model output is not a single letter -> cascade LLM judge
            question = kwargs.get("question", "")
            score, detail = cascade_answer_judge(
                ground_truth=gt,
                prediction=pred,
                question=question,
            )
            return {"correct": score >= 0.5, "score": score,
                    "detail": f"cascade_judge: {detail}"}

        # Open-ended answer: fuzzy matching
        score = fuzz.token_set_ratio(pred.lower(), gt.lower()) / 100.0
        correct = score >= 0.8
        return {"correct": correct, "score": score, "detail": f"fuzzy_score={score:.2f}"}
