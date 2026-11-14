
from omegaconf import DictConfig

from verl.trainer.ppo.core_algos import AdvantageEstimator

def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic"""
    if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    elif config.algorithm.adv_estimator in [
        AdvantageEstimator.GRPO,
        AdvantageEstimator.GRPO_PASSK,
        AdvantageEstimator.REINFORCE_PLUS_PLUS,
        # AdvantageEstimator.REMAX, # TODO:REMAX advantage estimator is not yet supported in one_step_off_policy
        AdvantageEstimator.RLOO,
        AdvantageEstimator.OPO,
        AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        AdvantageEstimator.GPG,
    ]:
        return False
    else:
        raise NotImplementedError
