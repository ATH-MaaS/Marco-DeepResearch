"""
The base class for reward model
"""

from abc import ABC, abstractmethod

from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.workers.config import HFModelConfig, RewardModelConfig

__all__ = ["BasePPORewardModel"]

class BasePPORewardModel(ABC):
    """base class for reward model"""

    def __init__(
        self,
        config: RewardModelConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh

    @abstractmethod
    def compute_reward(self, data: DataProto) -> DataProto:
        """Computing reward given input_ids. The transformers should output a tensor with shape
           [batch_size, sequence_length], and the value at [EOS] mask should be gathered.

        Args:
            data: must contain keys "input_ids", "attention_mask" and "position_ids".
                - input_ids: [batch_size, sequence_length]
                - attention_mask: [batch_size, sequence_length]
                - position_ids: [batch_size, sequence_length]

        Returns: a data pass protocol containing "reward". Only the [EOS] position contains the reward.
            Other position should have zero reward. Note that this may change in the future if we use
            dense reward. So, we leave the interface for general case.
            - reward: [batch_size, sequence_length].

        """
        pass
