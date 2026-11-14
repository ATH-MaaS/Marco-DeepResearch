
from importlib.metadata import PackageNotFoundError, version

from packaging import version as vs

from verl.utils.device import is_npu_available
from verl.utils.import_utils import is_sglang_available

def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None

package_name = "vllm"
package_version = get_version(package_name)
vllm_version = None
VLLM_SLEEP_LEVEL = 1

if package_version is None:
    if not is_sglang_available():
        raise ValueError(
            f"vllm version {package_version} not supported and SGLang also not Found. Currently supported "
            f"vllm versions are 0.7.0+"
        )
elif is_npu_available:
    # sleep_mode=2 is not supported on vllm-ascend for now, will remove this restriction when this ability is ready.
    VLLM_SLEEP_LEVEL = 1
    from vllm import LLM
    from vllm.distributed import parallel_state
elif vs.parse(package_version) >= vs.parse("0.7.0"):
    vllm_version = package_version
    if vs.parse(package_version) >= vs.parse("0.8.5"):
        VLLM_SLEEP_LEVEL = 2
    from vllm import LLM
    from vllm.distributed import parallel_state
else:
    if vs.parse(package_version) in [vs.parse("0.5.4"), vs.parse("0.6.3")]:
        raise ValueError(
            f"vLLM version {package_version} support has been removed. vLLM 0.5.4 and 0.6.3 are no longer "
            f"supported. Please use vLLM 0.7.0 or later."
        )
    if not is_sglang_available():
        raise ValueError(
            f"vllm version {package_version} not supported and SGLang also not Found. Currently supported "
            f"vllm versions are 0.7.0+"
        )

__all__ = ["LLM", "parallel_state"]
