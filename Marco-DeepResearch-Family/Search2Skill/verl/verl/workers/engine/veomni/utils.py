
import torch

from verl.utils.device import get_device_id, get_torch_device

VL_TYPE2INDEX = {
    "qwen2_5_vl": {
        "IMAGE_INPUT_INDEX": 151655,
        "VIDEO_INPUT_INDEX": 151656,
    },
    "qwen3_vl": {
        "IMAGE_INPUT_INDEX": 151655,
        "VIDEO_INPUT_INDEX": 151656,
    },
    "qwen3_vl_moe": {
        "IMAGE_INPUT_INDEX": 151655,
        "VIDEO_INPUT_INDEX": 151656,
    },
}

@torch.no_grad()
def offload_veomni_model_to_cpu(model, empty_cache: bool = True):
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    for module in model.modules():
        state = _get_module_fsdp_state(module)
        if state is None:
            continue
        fsdp_param_group = state._fsdp_param_group

        if fsdp_param_group is None:
            continue

        fsdp_param_group._training_state = TrainingState.IDLE

    model.reshard()
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()

@torch.no_grad()
def load_veomni_model_to_gpu(model):
    device = get_device_id()
    model.to(device)

@torch.no_grad()
def offload_veomni_optimizer(optimizer):
    optimizers = []
    # Check if this is a MultiOptimizer (for ep and non-ep parameters when ep+fsdp2 is enabled)
    if hasattr(optimizer, "_is_multi_optimizer") and optimizer._is_multi_optimizer:
        optimizers.extend(optimizer.optimizers_dict.values())
    else:
        optimizers.append(optimizer)

    for opt in optimizers:
        if not opt.state:
            continue
        for param_group in opt.param_groups:
            for param in param_group["params"]:
                state = opt.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)

@torch.no_grad()
def load_veomni_optimizer(optimizer, device_id):
    optimizers = []
    # Check if this is a MultiOptimizer (for ep and non-ep parameters when ep+fsdp2 is enabled)
    if hasattr(optimizer, "_is_multi_optimizer") and optimizer._is_multi_optimizer:
        optimizers.extend(optimizer.optimizers_dict.values())
    else:
        optimizers.append(optimizer)

    for opt in optimizers:
        if not opt.state:
            continue
        for param_group in opt.param_groups:
            for param in param_group["params"]:
                state = opt.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device_id, non_blocking=True)

def _map_moe_params_qwen3_moe(name, tensor):
    for i in range(tensor.size(0)):
        new_key = name.replace("mlp.experts.", f"mlp.experts.{i}.") + ".weight"
        yield new_key, tensor[i].to(get_device_id(), non_blocking=True)

MOE_PARAM_HANDERS = {
    "qwen3_moe": _map_moe_params_qwen3_moe,
}
