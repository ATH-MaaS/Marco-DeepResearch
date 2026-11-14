import re
import random
from typing import List, Dict
from datasets import load_dataset
from .base import BaseBenchmark


class FinanceBenchBenchmark(BaseBenchmark):
    """FinanceBench: open-ended financial QA."""

    def load(self, data_dir: str, num_samples: int = -1, subset: str = None, seed: int = None) -> List[Dict]:
        ds = load_dataset("PatronusAI/financebench", split="train",
                          cache_dir=data_dir)

        items = []
        for row in ds:
            items.append({
                "id": str(len(items)),
                "question": row.get("question", ""),
                "options": [],
                "context": row.get("evidence", ""),
                "ground_truth": row.get("answer", ""),
                "domain": "finance",
                "metadata": {
                    "company": row.get("company", ""),
                    "doc_name": row.get("doc_name", ""),
                },
            })

        if num_samples > 0 and num_samples < len(items):
            rng = random.Random(seed) if seed is not None else random.Random()
            items = rng.sample(items, num_samples)

        return items

    def format_question(self, item: Dict) -> str:
        q = item["question"]
        if item.get("context"):
            q += f"\n\nReference information:\n{item['context']}"
        return q

    def get_ground_truth(self, item: Dict) -> str:
        return item["ground_truth"]

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> Dict:
        pred = prediction.strip().lower()
        gt = ground_truth.strip().lower()

        # Exact match
        if pred == gt:
            return {"correct": True, "score": 1.0, "detail": "exact_match"}

        # Numerical match
        pred_nums = re.findall(r"[-+]?\d*\.?\d+", pred)
        gt_nums = re.findall(r"[-+]?\d*\.?\d+", gt)
        if pred_nums and gt_nums:
            try:
                for pn in pred_nums:
                    for gn in gt_nums:
                        if abs(float(pn) - float(gn)) / max(abs(float(gn)), 1e-9) < 0.05:
                            return {"correct": True, "score": 1.0, "detail": f"num_match: {pn}~={gn}"}
            except ValueError:
                pass

        # Keyword overlap
        gt_words = set(gt.split())
        pred_words = set(pred.split())
        overlap = len(gt_words & pred_words) / max(len(gt_words), 1)
        if overlap > 0.6:
            return {"correct": True, "score": overlap, "detail": f"keyword_overlap={overlap:.2f}"}

        return {"correct": False, "score": 0.0, "detail": "no_match"}
