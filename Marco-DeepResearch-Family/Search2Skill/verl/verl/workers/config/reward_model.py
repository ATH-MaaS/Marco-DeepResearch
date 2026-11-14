
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

from .model import HFModelConfig
from .rollout import RolloutConfig

__all__ = ["SandboxFusionConfig", "RewardModelConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

@dataclass
class SandboxFusionConfig(BaseConfig):
    """Configuration for cloud/local sandbox fusion.

    Args:
        url (Optional[str]): Cloud/local function URL for sandbox execution.
        max_concurrent (int): Max concurrent requests allowed to sandbox.
        memory_limit_mb (int): Max memory limit for each sandbox process in MB.
    """

    url: Optional[str] = None
    max_concurrent: int = 64
    memory_limit_mb: int = 1024

@dataclass
class RewardModelConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields

    reward_manager: Optional[str] = None

    enable: bool = False
    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0

    # reward model args
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    model: HFModelConfig = field(default_factory=HFModelConfig)
    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.reward_manager is not None:
            logger.warning(
                f"`reward_model.reward_manager` is deprecated, but got value {self.reward_manager}. "
                "Please use `reward_manager.name instead. "
                "See `verl/trainer/config/config.py:RewardManagerConfig` for more details."
            )
