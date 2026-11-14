
from verl.utils import ray_compat as ray

@ray.remote
class TestWorker:
    def __init__(self, rank, world_size, group_name):
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        self.communicator = None

    def init(self):
        from verl.utils.rendezvous.ray_backend import create_nccl_communicator_in_ray

        self.communicator = create_nccl_communicator_in_ray(self.rank, self.world_size, self.group_name)

    def test(self):
        if self.communicator is None:
            return None
        return self.communicator.rank_id()

def test_rvdz():
    ray.init()

    group_name = "test_group"
    world_size = 2

    workers = [TestWorker.options(num_gpus=1).remote(rank, world_size, group_name) for rank in range(world_size)]

    ray.get([worker.init.remote() for worker in workers])

    ranks = ray.get([worker.test.remote() for worker in workers])

    assert ranks == [0, 1], f"expecting [0, 1], got {ranks}"

    ray.shutdown()
