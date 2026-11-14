import re
from typing import Dict
from rapidfuzz import fuzz


class Evaluator:
    """Answer evaluator."""

    def __init__(self, method: str = "exact_and_fuzzy", fuzzy_threshold: float = 0.8):
        self.method = method
        self.fuzzy_threshold = fuzzy_threshold

    @classmethod
    def from_config(cls, config) -> "Evaluator":
        e = config["evaluation"]
        return cls(method=e["method"], fuzzy_threshold=e["fuzzy_threshold"])

    def evaluate(self, prediction: str, ground_truth: str) -> Dict:
        if not prediction:
            return {"correct": False, "score": 0.0, "detail": "empty_prediction"}

        pred = prediction.strip()
        gt = ground_truth.strip()

        # Exact match
        if self._normalize(pred) == self._normalize(gt):
            return {"correct": True, "score": 1.0, "detail": "exact_match"}

        # Numerical match
        num_result = self._numerical_match(pred, gt)
        if num_result:
            return num_result

        # Fuzzy match
        score = fuzz.token_set_ratio(pred.lower(), gt.lower()) / 100.0
        if score >= self.fuzzy_threshold:
            return {"correct": True, "score": score, "detail": f"fuzzy_match={score:.2f}"}

        # Containment match (gt appears in pred)
        if self._normalize(gt) in self._normalize(pred):
            return {"correct": True, "score": 0.9, "detail": "contains_match"}

        return {"correct": False, "score": score, "detail": f"no_match, fuzzy={score:.2f}"}

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def _numerical_match(self, pred: str, gt: str, tolerance: float = 0.05) -> Dict | None:
        gt_nums = re.findall(r"[-+]?\d*\.?\d+", gt)
        if not gt_nums:
            return None
        pred_nums = re.findall(r"[-+]?\d*\.?\d+", pred)
        if not pred_nums:
            return None
        for gn in gt_nums:
            for pn in pred_nums:
                try:
                    gv, pv = float(gn), float(pn)
                    if abs(gv) < 1e-9:
                        if abs(pv) < 1e-9:
                            return {"correct": True, "score": 1.0, "detail": f"num_match: {pn}=={gn}"}
                    elif abs(gv - pv) / abs(gv) < tolerance:
                        return {"correct": True, "score": 1.0, "detail": f"num_match: {pn}~={gn}"}
                except ValueError:
                    continue
        return None
