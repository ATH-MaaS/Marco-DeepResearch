from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseBenchmark(ABC):
    """Unified benchmark interface."""

    @abstractmethod
    def load(self, data_dir: str, num_samples: int = -1, subset: str = None, seed: int = None) -> List[Dict]:
        """Load data and return a list of items in unified format.
        num_samples=-1 means load all.
        """
        pass

    @abstractmethod
    def format_question(self, item: Dict) -> str:
        """Format raw data into prompt input text."""
        pass

    @abstractmethod
    def get_ground_truth(self, item: Dict) -> str:
        """Get the reference answer."""
        pass

    @abstractmethod
    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> Dict:
        """Evaluate the predicted answer.
        Returns: {correct: bool, score: float, detail: str}

        Subclasses that need extra context (e.g. the original question for
        LLM-based judging) can accept them via **kwargs.
        """
        pass
