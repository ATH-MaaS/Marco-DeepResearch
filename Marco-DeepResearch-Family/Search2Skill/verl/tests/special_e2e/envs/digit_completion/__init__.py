
from transformers import AutoTokenizer, LlamaConfig

from .task import DigitCompletion, generate_ground_truth_response
from .tokenizer import CharTokenizer

AutoTokenizer.register(LlamaConfig, CharTokenizer, exist_ok=True)

__all__ = ["DigitCompletion", "generate_ground_truth_response", "CharTokenizer"]
