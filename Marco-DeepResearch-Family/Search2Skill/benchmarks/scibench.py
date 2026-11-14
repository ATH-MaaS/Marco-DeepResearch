import random
from typing import List, Dict
from datasets import load_dataset
from .base import BaseBenchmark
from evaluation.cascade_judge import cascade_answer_judge


class SciBenchBenchmark(BaseBenchmark):
    """SciBench: college-level scientific problem solving.

    Dataset: xw27/scibench, split="train" (692 questions).
    Covers physics, chemistry, math from textbooks (Atkins, etc.).
    Answers are numeric (answer_number field).
    """

    def load(self, data_dir: str, num_samples: int = -1,
             subset: str = None, seed: int = None) -> List[Dict]:
        ds = load_dataset("xw27/scibench", split="train",
                          cache_dir=data_dir)

        items = []
        for row in ds:
            source = row.get("source", "unknown")
            if subset and source.lower() != subset.lower():
                continue

            problem = row.get("problem_text", "")
            answer = row.get("answer_number", "").strip()
            unit = row.get("unit", "").strip()

            items.append({
                "id": str(len(items)),
                "question": problem,
                "options": [],
                "context": "",
                "ground_truth": answer,
                "domain": source,
                "metadata": {
                    "subfield": source,
                    "difficulty": "",
                    "unit": unit,
                    "answer_latex": row.get("answer_latex", ""),
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
        score, detail = cascade_answer_judge(
            ground_truth=ground_truth,
            prediction=prediction,
            question=kwargs.get("question", ""),
        )
        return {"correct": score >= 0.5, "score": score,
                "detail": detail}
