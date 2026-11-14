
# VANILLA_MBRIDGE
try:
    from verl.models.mcore.patch import apply_patch_mbridge

    apply_patch_mbridge()
    from mbridge import AutoBridge
    from mbridge.utils.post_creation_callbacks import freeze_moe_router, make_value_model
except ImportError:
    print("mbridge package not found. Please install mbridge with `pip install verl[mcore]` or `pip install mbridge`")
    raise

__all__ = ["AutoBridge", "make_value_model", "freeze_moe_router"]
