
from ..device import is_npu_available
from ..import_utils import is_nvtx_available
from .performance import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from .profile import DistProfiler, DistProfilerExtension, ProfilerConfig

# Select marker implementations by availability, but keep DistProfiler as our dispatcher
if is_nvtx_available():
    from .nvtx_profile import mark_annotate, mark_end_range, mark_start_range, marked_timer
elif is_npu_available:
    from .mstx_profile import mark_annotate, mark_end_range, mark_start_range, marked_timer
else:
    from .performance import marked_timer
    from .profile import mark_annotate, mark_end_range, mark_start_range

__all__ = [
    "GPUMemoryLogger",
    "log_gpu_memory_usage",
    "mark_start_range",
    "mark_end_range",
    "mark_annotate",
    "DistProfiler",
    "DistProfilerExtension",
    "ProfilerConfig",
    "simple_timer",
    "marked_timer",
]
