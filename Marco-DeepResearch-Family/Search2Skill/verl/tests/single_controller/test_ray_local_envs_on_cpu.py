"""
e2e test verl.single_controller.ray
"""

import os

from verl.utils import ray_compat as ray

from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

@ray.remote
class TestActor(Worker):
    def __init__(self) -> None:
        super().__init__()

    def getenv(self, key):
        val = os.getenv(key, f"{key} not set")
        return val

def test_basics():
    ray.init(num_cpus=100)

    # create 4 workers, each hold a GPU
    resource_pool = RayResourcePool([4], use_gpu=False)
    class_with_args = RayClassWithInitArgs(cls=TestActor)

    worker_group = RayWorkerGroup(
        resource_pool=resource_pool, ray_cls_with_init=class_with_args, name_prefix="worker_group_basic"
    )

    output = worker_group.execute_all_sync("getenv", key="RAY_LOCAL_WORLD_SIZE")
    assert output == ["4", "4", "4", "4"]

    ray.shutdown()

def test_customized_worker_env():
    ray.init(num_cpus=100)

    # create 4 workers, each hold a GPU
    resource_pool = RayResourcePool([4], use_gpu=False)
    class_with_args = RayClassWithInitArgs(cls=TestActor)

    worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=class_with_args,
        name_prefix="worker_group_customized",
        worker_env={
            "test_key": "test_value",  # new key will be appended
        },
    )

    output = worker_group.execute_all_sync("getenv", key="test_key")
    assert output == ["test_value", "test_value", "test_value", "test_value"]

    try:
        worker_group = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=class_with_args,
            name_prefix="worker_group_error",
            worker_env={
                "WORLD_SIZE": "100",  # override system env will result in error
            },
        )
    except ValueError as e:
        assert "WORLD_SIZE" in str(e)
    else:
        raise ValueError("test failed")

    ray.shutdown()

if __name__ == "__main__":
    test_basics()
    test_customized_worker_env()
