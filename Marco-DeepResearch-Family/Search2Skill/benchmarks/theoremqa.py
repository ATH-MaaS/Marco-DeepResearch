import random
from typing import List, Dict
from datasets import load_dataset
from .base import BaseBenchmark
from evaluation.cascade_judge import cascade_answer_judge


class TheoremQABenchmark(BaseBenchmark):
    """TheoremQA: Theorem-driven university-level STEM QA.

    Dataset: TIGER-Lab/TheoremQA, split="test" (800 questions).
    Answer types: integer, float, bool, list of integer, list of float, option.
    Evaluation uses cascade_answer_judge for all types.
    """

    OPTION_LETTERS = "ABCDEFGHIJ"

    def load(self, data_dir: str, num_samples: int = -1,
             subset: str = None, seed: int = None) -> List[Dict]:
        ds = load_dataset("TIGER-Lab/TheoremQA", split="test",
                          cache_dir=data_dir)

        # Remove Picture column at arrow level to avoid PIL decoding errors.
        # Use the underlying arrow table to check for non-null Picture values.
        if "Picture" in ds.column_names:
            import pyarrow.compute as pc
            table = ds.data
            picture_col = table.column("Picture")
            # Keep rows where Picture is null (no image needed)
            mask = pc.is_null(picture_col)
            table = table.filter(mask)
            table = table.drop(["Picture"])
            from datasets import Dataset as HFDataset
            ds = HFDataset(table)

        items = []
        for row in ds:
            answer_type = row.get("Answer_type", "")
            question_text = row.get("Question", "")

            # Use answer_type as domain for subset filtering
            domain = answer_type if answer_type else "unknown"
            if subset and domain.lower() != subset.lower():
                continue

            answer = str(row.get("Answer", ""))

            items.append({
                "id": str(len(items)),
                "question": question_text,
                "options": [],
                "context": "",
                "ground_truth": answer,
                "domain": domain,
                "metadata": {
                    "subfield": answer_type,
                    "difficulty": "",
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
        pred = prediction.strip()
        gt = ground_truth.strip()

        # For simple numeric/bool answers, try exact match first
        if gt.lower() in ("true", "false"):
            pred_lower = pred.lower().strip(".")
            if pred_lower in ("true", "false", "yes", "no"):
                pred_bool = pred_lower in ("true", "yes")
                gt_bool = gt.lower() == "true"
                correct = pred_bool == gt_bool
                return {"correct": correct, "score": 1.0 if correct else 0.0,
                        "detail": f"bool_match: pred={pred_lower}, gt={gt}"}

        # Try numeric exact match
        try:
            gt_num = float(gt)
            pred_num = float(pred)
            # Allow small relative tolerance for floats
            if gt_num == 0:
                correct = abs(pred_num) < 1e-6
            else:
                correct = abs(pred_num - gt_num) / abs(gt_num) < 0.01
            if correct:
                return {"correct": True, "score": 1.0,
                        "detail": f"numeric_match: pred={pred_num}, gt={gt_num}"}
        except (ValueError, TypeError):
            pass

        # Fall through to cascade judge
        score, detail = cascade_answer_judge(
            ground_truth=gt,
            prediction=pred,
            question=kwargs.get("question", ""),
        )
        return {"correct": score >= 0.5, "score": score,
                "detail": f"cascade_judge: {detail}"}
