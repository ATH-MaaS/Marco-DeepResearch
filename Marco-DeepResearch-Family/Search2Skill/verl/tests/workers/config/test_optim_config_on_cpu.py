
import pytest

from verl.workers.config.optimizer import FSDPOptimizerConfig

class TestFSDPOptimizerConfigCPU:
    def test_default_configuration(self):
        config = FSDPOptimizerConfig(lr=0.1)
        assert config.min_lr_ratio is None
        assert config.lr_scheduler_type == "constant"
        assert config.num_cycles == 0.5

    @pytest.mark.parametrize("lr_scheduler_type", ["constant", "cosine"])
    def test_valid_lr_scheduler_types(self, lr_scheduler_type):
        config = FSDPOptimizerConfig(lr_scheduler_type=lr_scheduler_type, lr=0.1)
        assert config.lr_scheduler_type == lr_scheduler_type

    @pytest.mark.parametrize("warmup_style", ["constant", "cosine"])
    def test_valid_warmup_style_types(self, warmup_style):
        config = FSDPOptimizerConfig(warmup_style=warmup_style, lr=0.1)
        assert config.lr_scheduler_type == warmup_style

    def test_invalid_lr_scheduler_type(self):
        with pytest.raises((ValueError, AssertionError)):
            FSDPOptimizerConfig(lr_scheduler_type="invalid_style", lr=0.1)

    def test_invalid_warmup_style_type(self):
        with pytest.raises((ValueError, AssertionError)):
            FSDPOptimizerConfig(warmup_style="invalid_style", lr=0.1)

    @pytest.mark.parametrize("num_cycles", [0.1, 1.0, 2.5])
    def test_num_cycles_configuration(self, num_cycles):
        config = FSDPOptimizerConfig(num_cycles=num_cycles, lr=0.1)
        assert config.num_cycles == num_cycles
