import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*warp.context.*")
warnings.filterwarnings("ignore", message=".*warp.math.*")

import hydra
from omegaconf import DictConfig, OmegaConf

from okapi.gymnasium.ppo_vision import PPOVisionConfig, train


def huzzah(cfg):
    OKAPI_LOGO = r"""
          JJJJJ     JJJJJ
    JJJJJJ:::::JJJJJ:::::JJJJJJ
    JJJJJJJ:::::JJJ::::::JJJJJJJ                666                                 6666
    UUUUUUJJ:::::JJ:::::JJUUUUUU                666                                     
    zzzzzzzzzJJ::::::JJzzzzzzzzz    666666666   666  6666    66666666   666666666    666
    zzzzzzzzJ:::::::::YJzzzzzzzz   666    6666  666666       66666666   666    666   666
    zzzzzzzJJ::JJ::JJ::Jzzzzzzzz   6666   666   666  6666  6666   666   666   6666   666
    zzzzzzzzJ::::::::::Jzzzzzzzz     666666     666    666   66666666   66666666     666
    zzzzzzzzJJ::::::::Jzzzzzzzzz                                        666             
    zzzzzzzzzzJ::JJ:JJzzzzzzzzz
        JJJJJJJJJ::JJJJJJJJJ
    """
    print(OKAPI_LOGO)
    print("\n" + "=" * 54)
    print(f"  Algorithm: \t\tPPO")
    print(f"  Environment: \t\t{cfg.env_id}")
    print(f"  # envs: \t\t{cfg.num_envs:,}")
    print(f"  # timesteps: \t\t{cfg.total_timesteps:,}")
    print(f"  # epochs: \t\t{cfg.num_iterations:,}")
    print(f"  Random Seed: \t\t{cfg.seed:,}")
    print("=" * 54 + "\n")


# ---------------------------
# Hydra entry point
# ---------------------------
@hydra.main(
    version_base=None, config_path="../../configs", config_name="ppo_gymnasium_vision"
)
def main(cfg: DictConfig):
    ppo_vision_cfg = PPOVisionConfig(**OmegaConf.to_container(cfg, resolve=True))
    huzzah(ppo_vision_cfg)
    train(ppo_vision_cfg)


if __name__ == "__main__":
    main()
