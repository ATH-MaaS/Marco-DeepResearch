from .base import BaseEngine, EngineRegistry
from .fsdp import FSDPEngine, FSDPEngineWithLMHead

__all__ = [
    "BaseEngine",
    "EngineRegistry",
    "FSDPEngine",
    "FSDPEngineWithLMHead",
]

try:
    from .veomni import VeOmniEngine, VeOmniEngineWithLMHead

    __all__ += ["VeOmniEngine", "VeOmniEngineWithLMHead"]
except ImportError:
    VeOmniEngine = None
    VeOmniEngineWithLMHead = None

# Mindspeed must be imported before Megatron to ensure the related monkey patches take effect as expected
try:
    from .mindspeed import MindspeedEngineWithLMHead

    __all__ += ["MindspeedEngineWithLMHead"]
except ImportError:
    MindspeedEngineWithLMHead = None

try:
    from .megatron import MegatronEngine, MegatronEngineWithLMHead

    __all__ += ["MegatronEngine", "MegatronEngineWithLMHead"]
except ImportError:
    MegatronEngine = None
    MegatronEngineWithLMHead = None
