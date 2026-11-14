import json
import random
from typing import List, Dict
from .base import BaseBenchmark


class PutnamAxiomBenchmark(BaseBenchmark):
    """Putnam-AXIOM: Putnam competition problems with numerical variations.

    Data format (JSONL):
        {"id": "B5_var1", "question": "...", "ground_truth": "...",
         "answer_type": "numerical", "domain": "mathematics", ...}

    Evaluation uses cascade_answer_judge (LLM-based).
    """

    def load(self, data_dir: str, num_samples: int = -1,
             subset: str = None, seed: int = None) -> List[Dict]:
        import os
        if data_dir.endswith(".jsonl"):
            file_path = data_dir
        else:
            file_path = os.path.join(data_dir, "variations.jsonl")

        items: List[Dict] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                meta = raw.get("metadata", {})
                items.append({
                    "id": raw["id"],
                    "question": raw["question"],
                    "options": [],
                    "context": "",
                    "ground_truth": raw.get("ground_truth", ""),
                    "domain": raw.get("domain", "mathematics"),
                    "metadata": {
                        "subfield": meta.get("subfield", "Putnam"),
                        "difficulty": meta.get("difficulty", "hard"),
                        "answer_type": raw.get("answer_type", ""),
                    },
                })

        if subset:
            items = [it for it in items
                     if it.get("metadata", {}).get("answer_type", "").lower() == subset.lower()]

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
