"""Lightweight Ray API stub (no external ray dependency)."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ObjectRef:
    def __init__(self, value: Any = None):
        self._value = value

    def remote(self, *args, **kwargs):
        return self


class ActorHandle:
    def __getattr__(self, name: str):
        def _remote(*args, **kwargs):
            return ObjectRef()

        _remote.remote = _remote
        return _remote


class _RemoteMethod:
    def __init__(self, concurrency_group: str = ""):
        self.concurrency_group = concurrency_group

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        return func


def method(*args, **kwargs):
    return _RemoteMethod(**kwargs)


class _RuntimeContext:
    def get_node_id(self) -> str:
        return "local-node"

    def get_accelerator_ids(self) -> dict[str, list[str]]:
        return {"GPU": ["0"]}


def remote(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(obj):
        if isinstance(obj, type):

            class _ActorWrapper(obj):
                pass

            _ActorWrapper.remote = lambda *a, **kw: ActorHandle()
            return _ActorWrapper

        obj.remote = lambda *a, **kw: ObjectRef()
        return obj

    return decorator


def init(*args, **kwargs):
    return None


def shutdown(*args, **kwargs):
    return None


def is_initialized() -> bool:
    return False


def get(obj, *args, **kwargs):
    if isinstance(obj, ObjectRef):
        return obj._value
    if isinstance(obj, list):
        return [get(item) for item in obj]
    return obj


def put(value, *args, **kwargs):
    return ObjectRef(value)


def wait(obj_refs, *args, **kwargs):
    if isinstance(obj_refs, list):
        return obj_refs, []
    return [obj_refs], []


def get_actor(*args, **kwargs):
    return ActorHandle()


def get_gpu_ids() -> list[int]:
    return [0]


def nodes() -> list[dict[str, Any]]:
    return [{"NodeID": "local-node", "NodeManagerAddress": "127.0.0.1", "Alive": True, "Resources": {"CPU": 1}}]


def timeline(*args, **kwargs):
    return None


def get_runtime_context() -> _RuntimeContext:
    return _RuntimeContext()


from verl.utils.ray_compat import actor as actor
from verl.utils.ray_compat import util as util
from verl.utils.ray_compat import _private as _private

util.get_node_ip_address = lambda: "127.0.0.1"

__all__ = [
    "ObjectRef",
    "ActorHandle",
    "method",
    "remote",
    "init",
    "shutdown",
    "is_initialized",
    "get",
    "put",
    "wait",
    "get_actor",
    "get_gpu_ids",
    "nodes",
    "timeline",
    "get_runtime_context",
    "actor",
    "util",
    "_private",
]
