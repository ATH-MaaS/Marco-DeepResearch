
import torch
from megatron.core.distributed import DistributedDataParallel as DDP

@torch.no_grad()
def copy_megatron_model_to_cpu(models):
    """
    Copy Megatron model parameters to CPU memory (non-destructive copy).
    Unlike offload_megatron_model_to_cpu which moves data, this function creates
    independent copies on CPU while keeping GPU data intact.

    Args:
        models: List of model chunks (DDP-wrapped or unwrapped)

    Returns:
        dict: CPU state containing copied parameters and buffers
    """
    cpu_state = {}

    for model_idx, model_chunk in enumerate(models):
        if isinstance(model_chunk, DDP):
            # Handle DDP-wrapped models
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            buffer_states = []

            for buffers in model_chunk_all_buffers:
                buffer_list = []
                for buffer in buffers:
                    buffer_state = {}

                    # Copy parameter data to CPU
                    if buffer.param_data.storage().size() > 0:
                        buffer_state["param_data"] = buffer.param_data.data.cpu().clone().pin_memory()

                    buffer_list.append(buffer_state)
                buffer_states.append(buffer_list)

            cpu_state[f"model_chunk_{model_idx}"] = {"buffer_states": buffer_states, "is_ddp": True}
        else:
            # Handle non-DDP models (ref module)
            model_state = {}
            for name, param in model_chunk.named_parameters():
                param_state = {"data": param.data.cpu().clone().pin_memory()}
                model_state[name] = param_state

            cpu_state[f"model_chunk_{model_idx}"] = {"model_state": model_state, "is_ddp": False}

    return cpu_state

@torch.no_grad()
def restore_megatron_model_from_cpu(models, cpu_state):
    """
    Restore Megatron model parameters from CPU memory back to GPU.

    Args:
        models: List of model chunks to restore to
        cpu_state: CPU state dict returned from copy_megatron_model_to_cpu
    """
    for model_idx, model_chunk in enumerate(models):
        chunk_key = f"model_chunk_{model_idx}"
        if chunk_key not in cpu_state:
            continue

        chunk_state = cpu_state[chunk_key]

        if chunk_state["is_ddp"] and isinstance(model_chunk, DDP):
            # Restore DDP buffers
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            buffer_states = chunk_state["buffer_states"]

            for buffers, buffer_list in zip(model_chunk_all_buffers, buffer_states, strict=False):
                for buffer, buffer_state in zip(buffers, buffer_list, strict=False):
                    # Restore parameter data
                    if "param_data" in buffer_state:
                        buffer.param_data.data.copy_(buffer_state["param_data"].to(buffer.param_data.device))

        elif not chunk_state["is_ddp"] and not isinstance(model_chunk, DDP):
            # Restore non-DDP models
            model_state = chunk_state["model_state"]
            for name, param in model_chunk.named_parameters():
                if name in model_state:
                    param_state = model_state[name]
                    param.data.copy_(param_state["data"].to(param.device))
