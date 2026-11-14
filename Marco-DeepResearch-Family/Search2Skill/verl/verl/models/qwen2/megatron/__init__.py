
from .modeling_qwen2_megatron import (
    ParallelQwen2ForCausalLM,
    # rmpad with megatron
    ParallelQwen2ForCausalLMRmPad,
    # rmpad with megatron and pipeline parallelism
    ParallelQwen2ForCausalLMRmPadPP,
    ParallelQwen2ForValueRmPad,
    ParallelQwen2ForValueRmPadPP,
    # original model with megatron
    ParallelQwen2Model,
)

__all__ = [
    "ParallelQwen2ForCausalLM",
    "ParallelQwen2ForCausalLMRmPad",
    "ParallelQwen2ForCausalLMRmPadPP",
    "ParallelQwen2ForValueRmPad",
    "ParallelQwen2ForValueRmPadPP",
    "ParallelQwen2Model",
]
