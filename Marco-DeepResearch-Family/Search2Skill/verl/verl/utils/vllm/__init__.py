
from .utils import TensorLoRARequest, VLLMHijack, is_version_ge

# The contents of vllm/patch.py should not be imported here, because the contents of
# patch.py should be imported after the vllm LLM instance is created. Therefore,
# wait until you actually start using it before importing the contents of
# patch.py separately.

__all__ = [
    "TensorLoRARequest",
    "VLLMHijack",
    "is_version_ge",
]
