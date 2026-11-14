import json
import random
from typing import List, Dict
from .base import BaseBenchmark
from evaluation.cascade_judge import cascade_answer_judge


class RLTrainBenchmark(BaseBenchmark):
    """RL training dataset benchmark.

    Loads from a JSONL file where each line contains fields such as
    ``id``, ``question``, ``answer``, ``options``, ``domain``,
    ``usable``, and ``rl_type``.

    Only items with ``rl_type == "explore"`` and ``usable == True``
    are kept.  Evaluation is performed by the cascade LLM judge.
    """

    def load(
        self,
        data_dir: str,
        num_samples: int = -1,
        subset: str = None,
        seed: int = None,
    ) -> List[Dict]:
        file_path = data_dir
        items: List[Dict] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)

                if raw.get("rl_type") != "explore":
                    continue
                if not raw.get("usable", True):
                    continue
                if subset and raw.get("domain", "").lower() != subset.lower():
                    continue

                options = raw.get("options", [])
                if options and isinstance(options, list):
                    formatted_options = [
                        f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options) if opt
                    ]
                else:
                    formatted_options = []

                items.append(
                    {
                        "id": str(raw.get("id", len(items))),
                        "question": raw.get("question", ""),
                        "options": formatted_options,
                        "context": "",
                        "ground_truth": raw.get("answer", ""),
                        "domain": raw.get("domain", ""),
                        "metadata": {
                            "difficulty": raw.get("difficulty", ""),
                            "sample_type": raw.get("sample_type", ""),
                        },
                    }
                )

        if 0 < num_samples < len(items):
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
        score, detail = cascade_answer_judge(
            ground_truth=ground_truth,
            prediction=prediction,
            **kwargs,
        )
        return {
            "correct": score >= 0.5,
            "score": score,
            "detail": detail,
        }
