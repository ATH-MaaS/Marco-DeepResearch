
import numbers

import torch
from megatron.core import ModelParallelConfig
from torch import nn
from transformers import LlamaConfig

from verl.utils.megatron import sequence_parallel as sp_utils

class ParallelLlamaRMSNorm(nn.Module):
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        if isinstance(config.hidden_size, numbers.Integral):
            normalized_shape = (config.hidden_size,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.variance_epsilon = config.rms_norm_eps

        if megatron_config.sequence_parallel:
            sp_utils.mark_parameter_as_sequence_parallel(self.weight)

    def forward(self, hidden_states):
        from apex.normalization.fused_layer_norm import fused_rms_norm_affine

        return fused_rms_norm_affine(
            input=hidden_states,
            weight=self.weight,
            normalized_shape=self.normalized_shape,
            eps=self.variance_epsilon,
            memory_efficient=True,
        )
