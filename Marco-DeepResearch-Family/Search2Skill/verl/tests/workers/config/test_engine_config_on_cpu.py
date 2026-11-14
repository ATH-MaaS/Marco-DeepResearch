
import pytest

from verl.workers.config.engine import FSDPEngineConfig, McoreEngineConfig

class TestMcoreEngineConfig:
    def test_default_values(self):
        config = McoreEngineConfig()
        assert config.tensor_model_parallel_size == 1
        assert config.sequence_parallel is False  # Should be auto-corrected
        assert config.seed == 42

    def test_post_init_validation(self):
        # Test TP size 1 forces sequence_parallel=False
        config = McoreEngineConfig(tensor_model_parallel_size=1)
        assert config.sequence_parallel is False

        # Test TP >1 keeps sequence_parallel=True
        config = McoreEngineConfig(tensor_model_parallel_size=2)
        assert config.sequence_parallel is True

    def test_mutable_fields(self):
        config = McoreEngineConfig()
        config.sequence_parallel = True  # Should be mutable
        with pytest.raises(AttributeError):
            config.tensor_model_parallel_size = 2  # Frozen field

    @pytest.mark.parametrize("offload_field", ["param_offload", "grad_offload", "optimizer_offload"])
    def test_offload_flags(self, offload_field):
        config = McoreEngineConfig(**{offload_field: True})
        assert getattr(config, offload_field) is True

class TestFSDPEngineConfigCPU:
    def test_default_values(self):
        config = FSDPEngineConfig()
        assert config.param_offload is False
        assert config.optimizer_offload is False
        assert config.fsdp_size == -1

    @pytest.mark.parametrize(
        "offload_params",
        [{"param_offload": True}, {"optimizer_offload": True}, {"param_offload": True, "optimizer_offload": True}],
    )
    def test_offload_combinations(self, offload_params):
        config = FSDPEngineConfig(**offload_params)
        assert config.param_offload == offload_params.get("param_offload", False)
        assert config.optimizer_offload == offload_params.get("optimizer_offload", False)

    def test_wrap_policy_configuration(self):
        test_policy = {"layer_class": "TransformerBlock"}
        config = FSDPEngineConfig(wrap_policy=test_policy)
        assert config.wrap_policy == test_policy
