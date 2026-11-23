"""Train any okapi algorithm on a MuJoCo Playground environment.

uv run scripts/train.py algo=sac env_id=CheetahRun
uv run scripts/train.py algo=ppo env_id=Go1JoystickFlatTerrain value_obs_key=privileged_state
uv run scripts/train.py algo=simbav2 env_id=CheetahRun seed=1 use_wandb=true checkpoint=true
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import hydra
from omegaconf import DictConfig, OmegaConf

import okapi

OKAPI_LOGO = r"""
       JJJJJJJJJJJJJJJJJJJJ
   JJJ::JJJ::::::::::::JJJ::JJJ
    J:::::::JJ:JJJJ:JJ:::::::J              JJJ                             JJJ
    JJJ:::::::J::::J::J::::JJJ              JJJ                             JJJ
    JJJJJJ::J::::::::J::JJJJJJ              JJJ
    ::::::::J:::::::::J:::::::   JJJJJJJJ   JJJ JJJJ   JJJJJJJ   JJJJJJJJ   JJJ
    :::::::J::J::::JJ:J:::::::  JJJ   JJJ   JJJJJJ          JJ   JJJ   JJJ  JJJ
    JJJJJJJJ::::::::::JJJJJJJJ  JJJ    JJJ  JJJJJJ    JJJJJJJJ   JJJ   JJJ  JJJ
    JJJJJJJJf:::::::::JJJJJJJJ  JJJ   JJJ   JJJ JJJ   JJJ   JJ   JJJ   JJJ  JJJ
    JJJJJJJJJ::::::::JJJJJJJJJ    JJJJJJ    JJJ  JJJJ  JJJJJJJ   JJJJJJJJ   JJJ
    :::::::::J::::::J:::::::::                                   JJJ
     JJJJJJJJJJ:::::JJJJJJJJJ                                    JJJ
      JJJJJJJJJ::::JJJJJJJJJ
                JJJ
"""


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    values = OmegaConf.to_container(cfg, resolve=True)
    name = values.pop("algo")
    algo = okapi.get_algorithm(name)
    config = algo.Config.from_dict(values)
    if config.verbose:
        print(OKAPI_LOGO)
        print("=" * 54)
        print(f"  Algorithm:     {name}")
        print(f"  Environment:   {config.env_id}")
        print(f"  # envs:        {config.num_envs:,}")
        print(f"  # timesteps:   {config.total_timesteps:,}")
        print(f"  Seed:          {config.seed}")
        print("=" * 54 + "\n")
    algo.train(config)


if __name__ == "__main__":
    main()
