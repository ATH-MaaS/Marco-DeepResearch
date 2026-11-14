
from .modeling_llama_megatron import (
    ParallelLlamaForCausalLM,
    # rmpad with megatron
    ParallelLlamaForCausalLMRmPad,
    # rmpad with megatron and pipeline parallelism
    ParallelLlamaForCausalLMRmPadPP,
    ParallelLlamaForValueRmPad,
    ParallelLlamaForValueRmPadPP,
    # original model with megatron
    ParallelLlamaModel,
)

__all__ = [
    "ParallelLlamaForCausalLM",
    "ParallelLlamaForCausalLMRmPad",
    "ParallelLlamaForCausalLMRmPadPP",
    "ParallelLlamaForValueRmPad",
    "ParallelLlamaForValueRmPadPP",
    "ParallelLlamaModel",
]
