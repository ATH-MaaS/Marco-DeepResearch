
import logging
import os

try:
    from mindspeed.megatron_adaptor import repatch
except ImportError:
    repatch = None

from verl.trainer.config import CheckpointConfig
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig

from ..base import EngineRegistry
from ..megatron import MegatronEngineWithLMHead

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

@EngineRegistry.register(model_type="language_model", backend="megatron", device="npu")
class MindspeedEngineWithLMHead(MegatronEngineWithLMHead):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

        repatch_config = {"use_flash_attn": True}
        if self.engine_config.context_parallel_size > 1:
            repatch_config["context_parallel_size"] = self.engine_config.context_parallel_size

        repatch(repatch_config)
