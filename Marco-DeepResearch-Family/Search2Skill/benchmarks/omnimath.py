import json
import random
from typing import List, Dict
from .base import BaseBenchmark


class OmniMathBenchmark(BaseBenchmark):
    """OmniMath: competition-level open-ended math problems.

    Expected data format (JSONL, one JSON object per line):
        {
            "domain": ["Mathematics -> ..."],
            "difficulty": 6.5,
            "problem": "...",
            "solution": "...",
            "answer": "...",
            "source": "usamo",
            "_idx": 35
        }

    The ``answer`` field is a free-form math expression (not multiple-choice),
    so evaluation must use a judge (e.g. cascade_answer_judge) rather than
    simple string matching.
    """

    def load(self, data_dir: str, num_samples: int = -1,
             subset: str = None, seed: int = None) -> List[Dict]:
        """Load from a pre-extracted JSONL file supplied via --data_file.

        If ``data_dir`` points to a ``.jsonl`` file directly we read that;
        otherwise we fall back to a default path under ``data_dir``.
        """
        # When invoked through eval_benchmark.py with --data_file, the items
        # are loaded externally (JSON list).  This method handles the
        # HuggingFace / raw-file fallback path.
        import os
        if data_dir.endswith(".jsonl"):
            file_path = data_dir
        else:
            file_path = os.path.join(data_dir, "omnimath.jsonl")

        items: List[Dict] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                domain_list = raw.get("domain", [])
                domain = domain_list[0] if domain_list else "Mathematics"
                # Extract top-level domain label (e.g. "Algebra", "Combinatorics")
                domain_short = domain.split("->")[1].strip() if "->" in domain else domain

                if subset and domain_short.lower() != subset.lower():
                    continue

                diff_val = raw.get("difficulty", 0)
                items.append({
                    "id": str(raw.get("_idx", len(items))),
                    "question": raw["problem"],
                    "options": [],           # open-ended, no options
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
        """Evaluate using cascade_answer_judge for open-ended math answers.

        Falls back to normalized text matching when the judge is unavailable.
        The caller (eval_benchmark.py) is responsible for passing ``question``
        and optional judge profile overrides.
        """
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
