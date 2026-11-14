
from enum import Enum

import torch
from tensordict.tensorclass import NonTensorData

class DatasetPadMode(str, Enum):
    """Padding mode for dataset"""

    RIGHT = "right"
    LEFT_RIGHT = "left_right"
    NO_PADDING = "no_padding"

class SFTTensorCollator:
    """
    A custom collate_fn that handles batching of sequences.
    1. for variable-length sequences, convert them into NestedTensors.
    2. for fixed-length sequences, use default_collate.
    """

    def __init__(self, pad_mode: DatasetPadMode = DatasetPadMode.LEFT_RIGHT):
        self.pad_mode = pad_mode

    def __call__(self, batch: list[dict[str, any]]) -> dict[str, any]:
        if self.pad_mode == DatasetPadMode.NO_PADDING:
            return self.collate_variable_batch(batch)
        elif self.pad_mode in [DatasetPadMode.RIGHT, DatasetPadMode.LEFT_RIGHT]:
            from torch.utils.data import default_collate

            return default_collate(batch)
        else:
            raise NotImplementedError(f"pad_mode {self.pad_mode} not implemented")

    def collate_variable_batch(self, batch: list[dict[str, any]]) -> dict[str, any]:
        """
        Collates a list of samples into a single batch.

        Args:
            batch: A list of dictionary samples from the dataset.

        Returns:
            A dictionary representing the batched data, with variable-length
            sequences converted to NestedTensors.
        """

        final_batch = {}

        tensor_keys = set().union(*(d.keys() for d in batch))

        # Handle tensor values by creating a NestedTensor.
        for key in tensor_keys:
            if isinstance(batch[0][key], torch.Tensor):
                tensors = [item[key] for item in batch]
                final_batch[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
            else:
                tensors = [NonTensorData(item.get(key)) for item in batch]
                final_batch[key] = torch.stack(tensors, dim=0)

        return final_batch
