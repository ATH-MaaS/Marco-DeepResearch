import random
from typing import List, Dict
from datasets import load_dataset
from .base import BaseBenchmark


class MMLUProBenchmark(BaseBenchmark):
    """MMLU-Pro: multiple-choice with 10 options across 14 subjects."""

    OPTION_LETTERS = "ABCDEFGHIJ"

    def load(self, data_dir: str, num_samples: int = -1, subset: str = None, seed: int = None) -> List[Dict]:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", "default", split="test",
                          cache_dir=data_dir, )

        items = []
        for row in ds:
            # Filter by subset (subject) if specified
            if subset and row.get("category", "").lower() != subset.lower():
                continue
            options = row.get("options", [])
            formatted_options = [
                f"{self.OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
            ]
            items.append({
                "id": row.get("question_id", str(len(items))),
                "question": row["question"],
                "options": formatted_options,
                "context": "",
                "ground_truth": row.get("answer", ""),
                "domain": row.get("category", "general"),
                "metadata": {"src": row.get("src", "")},
            })

        if num_samples > 0 and num_samples < len(items):
            rng = random.Random(seed) if seed is not None else random.Random()
            items = rng.sample(items, num_samples)

        return items

    def format_question(self, item: Dict) -> str:
        options_text = "\n".join(item["options"])
        return f"{item['question']}\n\n{options_text}"

    def get_ground_truth(self, item: Dict) -> str:
        return item["ground_truth"]

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> Dict:
        # Extract answer letter from prediction
        pred_letter = self._extract_answer_letter(prediction)
        gt_letter = ground_truth.strip().upper()
        correct = pred_letter == gt_letter
        return {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "detail": f"pred={pred_letter}, gt={gt_letter}",
        }

    def _extract_answer_letter(self, text: str) -> str:
        text = text.strip().upper()
        # Single letter directly
        if len(text) == 1 and text in self.OPTION_LETTERS:
            return text
        # "The answer is X" pattern
        import re
        match = re.search(r"(?:answer|答案)\s*(?:is|为|：|:)?\s*([A-J])", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # First option letter found
        for ch in text:
            if ch in self.OPTION_LETTERS:
                return ch
        return ""
