
from verl.utils import ray_compat as ray

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
    create_colocated_worker_raw_cls,
)

@ray.remote
class Actor(Worker):
    def __init__(self) -> None:
        super().__init__()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def add(self, x):
        x += self.rank
        return x

@ray.remote
class Critic(Worker):
    def __init__(self, val) -> None:
        super().__init__()
        self.val = val

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def sub(self, x):
        x -= self.val
        return x

actor_cls = RayClassWithInitArgs(cls=Actor)
critic_cls = RayClassWithInitArgs(cls=Critic, val=10)
cls_dict = {"actor": actor_cls, "critic": critic_cls}
FusedBaseClass = create_colocated_worker_raw_cls(cls_dict)

@ray.remote
class HybridWorker(FusedBaseClass):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def foo(self, x):
        return self.critic.sub(self.actor.add(x))

def test_fused_workers():
    ray.init(num_cpus=100)

    # create separate workers on the same resource pool
    process_on_nodes = [2]
    resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=False)

    # create colocated workers
    hybrid_cls_with_init = RayClassWithInitArgs(cls=HybridWorker)
    hybrid_cls_with_init.fused_worker_used = True

    fused_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=hybrid_cls_with_init)
    fused_wg.fuse(cls_dict.keys())

    x = fused_wg.actor.add(0.1)
    print(x)
    y = fused_wg.critic.sub(x)
    print(y)
    z = fused_wg.foo(0.1)
    print(z)
    for i, j in zip(y, z, strict=True):
        assert i == j

    ray.shutdown()

if __name__ == "__main__":
    test_fused_workers()
