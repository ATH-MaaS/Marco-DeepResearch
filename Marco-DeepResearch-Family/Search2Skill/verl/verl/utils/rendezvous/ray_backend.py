
import logging
import time

from verl.utils import ray_compat as ray
from cupy.cuda.nccl import NcclCommunicator, get_unique_id
from verl.utils.ray_compat.util import list_named_actors

@ray.remote
class NCCLIDStore:
    def __init__(self, nccl_id):
        self._nccl_id = nccl_id

    def get(self):
        return self._nccl_id

def get_nccl_id_store_by_name(name):
    all_actors = list_named_actors(all_namespaces=True)
    matched_actors = [actor for actor in all_actors if actor.get("name", None) == name]
    if len(matched_actors) == 1:
        actor = matched_actors[0]
        return ray.get_actor(**actor)
    elif len(matched_actors) > 1:
        logging.warning("multiple actors with same name found: %s", matched_actors)
    elif len(matched_actors) == 0:
        logging.info("failed to get any actor named %s", name)
    return None

def create_nccl_communicator_in_ray(
    rank: int, world_size: int, group_name: str, max_retries: int = 100, interval_s: int = 5
):
    if rank == 0:
        nccl_id = get_unique_id()
        nccl_id_store = NCCLIDStore.options(name=group_name).remote(nccl_id)

        assert ray.get(nccl_id_store.get.remote()) == nccl_id
        communicator = NcclCommunicator(
            ndev=world_size,
            commId=nccl_id,
            rank=0,
        )
        return communicator
    else:
        for i in range(max_retries):
            nccl_id_store = get_nccl_id_store_by_name(group_name)
            if nccl_id_store is not None:
                logging.info("nccl_id_store %s got", group_name)
                nccl_id = ray.get(nccl_id_store.get.remote())
                logging.info("nccl id for %s got: %s", group_name, nccl_id)
                communicator = NcclCommunicator(
                    ndev=world_size,
                    commId=nccl_id,
                    rank=rank,
                )
                return communicator
            logging.info("failed to get nccl_id for %d time, sleep for %d seconds", i + 1, interval_s)
            time.sleep(interval_s)
