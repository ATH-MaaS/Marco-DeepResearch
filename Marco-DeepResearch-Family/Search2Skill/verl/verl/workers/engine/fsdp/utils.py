from torch.distributed.device_mesh import init_device_mesh

from verl.utils.device import get_device_name

def create_device_mesh(world_size, fsdp_size):
    """
    Create a device mesh for distributed training based on the world size and FSDP size.

    Args:
        world_size (int): Total number of processes in the distributed training setup.
        fsdp_size (int): Size of the Fully Sharded Data Parallel (FSDP) group.

    Returns:
        torch.distributed.device_mesh.DeviceMesh: The initialized device mesh.
    """
    device_name = get_device_name()
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh

def get_sharding_strategy(device_mesh):
    """
    Determine the appropriate sharding strategy based on the number of dimensions of the device mesh.

    Args:
        device_mesh (torch.distributed.device_mesh.DeviceMesh): The device mesh used for distributed training.

    Returns:
        torch.distributed.fsdp.ShardingStrategy: The sharding strategy to be used with FSDP.

    Raises:
        NotImplementedError: If the number of dimensions of the device mesh is neither 1 nor 2.
    """
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy
