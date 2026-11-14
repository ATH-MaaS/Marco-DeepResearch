"""
Base class for a critic
"""

from abc import ABC, abstractmethod

import torch

from verl import DataProto

__all__ = ["BasePPOCritic"]

class BasePPOCritic(ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_values(self, data: DataProto) -> torch.Tensor:
        """Compute values"""
        pass

    @abstractmethod
    def update_critic(self, data: DataProto):
        """Update the critic"""
        pass
