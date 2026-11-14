
import os

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["NCCL_DEBUG"] = "WARN"

from verl.utils import ray_compat as ray
import torch
import torch.distributed

from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.device import get_device_name

@ray.remote
class TestAllGatherActor(Worker):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    def init(self):
        torch.distributed.init_process_group()
        self.tensor = torch.zeros(size=(self.size,), dtype=torch.int64, device=get_device_name())
        self.tensor += self.rank

    def all_gather(self):
        world_size = self._world_size
        output = torch.zeros(
            size=(self.tensor.shape[0] * world_size,), dtype=self.tensor.dtype, device=self.tensor.device
        )
        torch.distributed.all_gather_into_tensor(output, self.tensor, async_op=False)
        return output

@ray.remote
class TestAllGatherActorV2(Worker):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

        torch.distributed.init_process_group()
        self.tensor = torch.zeros(size=(self.size,), dtype=torch.int64, device=get_device_name())
        self.tensor += self.rank

    def all_gather(self):
        world_size = self._world_size
        output = torch.zeros(
            size=(self.tensor.shape[0] * world_size,), dtype=self.tensor.dtype, device=self.tensor.device
        )
        torch.distributed.all_gather_into_tensor(output, self.tensor, async_op=False)
        return output

def test_all_gather_torch():
    """
    In this test, we instantiate 4 GPUs in a group and test the all_gather
    """
    ray.init()

    # create 4 workers, each hold a GPU
    resource_pool = RayResourcePool([4], use_gpu=True)
    class_with_args = RayClassWithInitArgs(cls=TestAllGatherActor, size=2)

    worker_group = RayWorkerGroup(
        resource_pool, class_with_args, name_prefix="worker_group_torch", device_name=get_device_name()
    )

    worker_group.execute_all_sync("init")
    output = worker_group.execute_all_sync("all_gather")
    for i in range(1, len(output)):
        assert torch.all(output[i] == output[0])

    output = output[0].cpu()
    print(output)
    assert torch.all(output == torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64))

    ray.shutdown()

def test_all_gather_torch_v2():
    """
    In this test, we instantiate 4 GPUs in a group and test the all_gather
    """
    ray.init()

    # create 4 workers, each hold a GPU
    resource_pool = RayResourcePool([4], use_gpu=True)
    class_with_args = RayClassWithInitArgs(cls=TestAllGatherActorV2, size=2)

    worker_group = RayWorkerGroup(
        resource_pool, class_with_args, name_prefix="worker_group_torch", device_name=get_device_name()
    )

    output = worker_group.execute_all_sync("all_gather")
    for i in range(1, len(output)):
        assert torch.all(output[i] == output[0])

    output = output[0].cpu()
    print(output)
    assert torch.all(output == torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64))

    ray.shutdown()
