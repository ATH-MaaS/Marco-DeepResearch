from verl.utils.ray_compat.util import collective as collective
from verl.utils.ray_compat.util import placement_group as placement_group
from verl.utils.ray_compat.util import scheduling_strategies as scheduling_strategies

get_node_ip_address = lambda: "127.0.0.1"
list_named_actors = lambda *args, **kwargs: []
placement_group_table = lambda *args, **kwargs: {}

__all__ = [
    "get_node_ip_address",
    "list_named_actors",
    "placement_group",
    "placement_group_table",
    "scheduling_strategies",
    "collective",
]
