import random
from typing import List, Dict
from datasets import load_dataset
from .base import BaseBenchmark
from evaluation.cascade_judge import cascade_answer_judge


class GPQADiamondBenchmark(BaseBenchmark):
    """GPQA Diamond: Graduate-level multiple-choice QA (Biology, Physics, Chemistry).

    Dataset: Idavidrein/gpqa, "gpqa_diamond" config.
    Each question has 4 options (1 correct + 3 incorrect).
    """

    OPTION_LETTERS = "ABCD"

    def load(self, data_dir: str, num_samples: int = -1,
             subset: str = None, seed: int = None) -> List[Dict]:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train",
                          cache_dir=data_dir)

        items = []
        for row in ds:
            domain = row.get("Subdomain", "general")
            if subset and domain.lower() != subset.lower():
                continue

            # Build options: shuffle correct + incorrect answers with fixed seed per question
            correct_answer = row.get("Correct Answer", "")
            incorrect = [
                row.get("Incorrect Answer 1", ""),
                row.get("Incorrect Answer 2", ""),
                row.get("Incorrect Answer 3", ""),
            ]
            all_answers = [correct_answer] + incorrect

            # Deterministic shuffle per question so answer position is fixed
            q_seed = hash(row.get("Question", "")) % (2**31)
            q_rng = random.Random(q_seed)
            q_rng.shuffle(all_answers)

            correct_idx = all_answers.index(correct_answer)
            correct_letter = self.OPTION_LETTERS[correct_idx]

            formatted_options = [
                f"{self.OPTION_LETTERS[i]}. {opt}"
                for i, opt in enumerate(all_answers) if opt
            ]

            items.append({
                "id": str(len(items)),
                "question": row.get("Question", ""),
                "options": formatted_options,
                "context": "",
                "ground_truth": correct_letter,
                "domain": domain,
                "metadata": {
                    "subfield": domain,
                    "difficulty": "",
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

        if len(gt) == 1 and gt.upper() in self.OPTION_LETTERS:
            if len(pred) <= 2 and pred and pred[0].upper() in self.OPTION_LETTERS:
                pred_letter = pred[0].upper()
                correct = pred_letter == gt.upper()
                return {"correct": correct, "score": 1.0 if correct else 0.0,
                        "detail": f"pred={pred_letter}, gt={gt}"}

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
