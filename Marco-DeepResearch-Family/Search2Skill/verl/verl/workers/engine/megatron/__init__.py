
# HACK Avoid cpu worker trigger cuda jit error
import os

from verl.utils.device import is_cuda_available

if not is_cuda_available and "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

from .transformer_impl import MegatronEngine, MegatronEngineWithLMHead  # noqa: E402

if not is_cuda_available:
    del os.environ["TORCH_CUDA_ARCH_LIST"]

__all__ = ["MegatronEngine", "MegatronEngineWithLMHead"]
