import json
import random
from typing import List, Dict
from .base import BaseBenchmark


class EvoAgentBenchBenchmark(BaseBenchmark):
    """EvoAgentBench: competition-level open-ended math problems.

    Expected data format (JSONL, one JSON object per line):
        {
            "domain": ["Mathematics -> ..."],
            "difficulty": 6.5,
            "problem": "...",
            "solution": "...",
            "answer": "...",
            "source": "...",
            "_idx": 35
        }

    Uses cascade_answer_judge for evaluation (same as OmniMath).
    """

    def load(self, data_dir: str, num_samples: int = -1,
             subset: str = None, seed: int = None) -> List[Dict]:
        import os
        if data_dir.endswith(".jsonl"):
            file_path = data_dir
        else:
            file_path = os.path.join(data_dir, "evoagentbench.jsonl")

        items: List[Dict] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                domain_list = raw.get("domain", [])
                domain = domain_list[0] if domain_list else "Mathematics"
                domain_short = domain.split("->")[1].strip() if "->" in domain else domain

                if subset and domain_short.lower() != subset.lower():
                    continue

                diff_val = raw.get("difficulty", 0)
                items.append({
                    "id": str(raw.get("_idx", len(items))),
                    "question": raw["problem"],
                    "options": [],
                    "context": "",
                    "ground_truth": raw.get("answer", ""),
                    "solution": raw.get("solution", ""),
                    "domain": domain_short,
                    "metadata": {
                        "subfield": domain,
                        "difficulty": str(diff_val),
                        "source": raw.get("source", ""),
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
