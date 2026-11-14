import random
from typing import List, Dict
from datasets import load_dataset
from .base import BaseBenchmark
from evaluation.cascade_judge import cascade_answer_judge


class ChemBench4KBenchmark(BaseBenchmark):
    """ChemBench4K: Chemistry multiple-choice QA benchmark.

    Dataset: AI4Chem/ChemBench4K, validation split only (45 questions).
    Each question has 4 options (A/B/C/D), answer is a single letter.
    """

    OPTION_LETTERS = "ABCD"

    def load(self, data_dir: str, num_samples: int = -1,
             subset: str = None, seed: int = None) -> List[Dict]:
        ds = load_dataset("AI4Chem/ChemBench4K", split="validation",
                          cache_dir=data_dir)

        items = []
        for row in ds:
            question = row.get("question", "")
            options = [
                f"A. {row.get('A', '')}",
                f"B. {row.get('B', '')}",
                f"C. {row.get('C', '')}",
                f"D. {row.get('D', '')}",
            ]
            ground_truth = row.get("answer", "").strip().upper()

            items.append({
                "id": str(len(items)),
                "question": question,
                "options": options,
                "context": "",
                "ground_truth": ground_truth,
                "domain": "Chemistry",
                "metadata": {
                    "subfield": "",
                    "difficulty": "",
                },
            })

        if subset:
            items = [it for it in items if it["domain"].lower() == subset.lower()]

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

        # Try direct letter match
        if len(gt) == 1 and gt in self.OPTION_LETTERS:
            if len(pred) <= 2 and pred and pred[0].upper() in self.OPTION_LETTERS:
                pred_letter = pred[0].upper()
                correct = pred_letter == gt
                return {"correct": correct, "score": 1.0 if correct else 0.0,
                        "detail": f"pred={pred_letter}, gt={gt}"}

            # Fallback to cascade judge
            question = kwargs.get("question", "")
            score, detail = cascade_answer_judge(
                ground_truth=gt,
                prediction=pred,
                question=question,
            )
            return {"correct": score >= 0.5, "score": score,
                    "detail": f"cascade_judge: {detail}"}

        score, detail = cascade_answer_judge(
            ground_truth=gt,
            prediction=pred,
            question=kwargs.get("question", ""),
        )
        return {"correct": score >= 0.5, "score": score,
                "detail": f"cascade_judge: {detail}"}
