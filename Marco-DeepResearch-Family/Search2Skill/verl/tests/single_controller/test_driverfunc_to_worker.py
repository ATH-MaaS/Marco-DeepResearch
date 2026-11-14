
import os

from verl.utils import ray_compat as ray
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray import RayWorkerGroup
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool
from verl.utils.device import get_device_name

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["NCCL_DEBUG"] = "WARN"

@ray.remote
class ModelActor(Worker):
    def __init__(self):
        pass

class HackSelf:
    def __init__(self):
        pass

def get_aux_metrics(self, test_proto):
    sequence_ids = test_proto.batch["sequence_ids"]
    decode_count = []
    for i in range(sequence_ids.size(0)):
        decode_count.append(len(sequence_ids[i].tolist()))
    ret_proto = DataProto(
        batch=TensorDict(
            {"sequence_ids": sequence_ids, "decode_count": torch.tensor(decode_count)}, batch_size=sequence_ids.size(0)
        )
    )
    return ret_proto

def test():
    # construct model
    ray.init()

    # create 2 workers, each hold a GPU
    resource_pool = RayResourcePool([2], use_gpu=True, name_prefix="a")

    class_with_args = RayClassWithInitArgs(cls=ModelActor)
    shard_wg = RayWorkerGroup(resource_pool, class_with_args, device_name=get_device_name())

    test_bs = 8
    test_proto = DataProto(
        TensorDict(
            {
                "sequence_ids": torch.ones([test_bs, 2048], dtype=torch.int64),
            },
            batch_size=test_bs,
        ),
        meta_info={"query_length": 1536},
    )

    # Sharding among different ranks
    ret_proto1 = shard_wg.execute_with_func_generator(get_aux_metrics, test_proto)

    # compare execute on driver
    hs = HackSelf()
    ret_proto2 = get_aux_metrics(hs, test_proto)

    torch.testing.assert_close(ret_proto1.batch["decode_count"], ret_proto2.batch["decode_count"])

    ray.shutdown()
