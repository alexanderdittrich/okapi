"""configs/algo/*.yaml must list exactly the options of each algorithm's config, with its defaults."""

from dataclasses import fields
from pathlib import Path

import pytest
import yaml

import okapi
from okapi.common.config import TrainConfig

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize("name", sorted(okapi.ALGORITHMS))
def test_algo_yaml_matches_dataclass(name):
    values = yaml.safe_load((CONFIGS / "algo" / f"{name}.yaml").read_text())
    assert values.pop("algo") == name
    config = okapi.get_algorithm(name).Config
    defaults = config()
    own = {f.name for f in fields(config)} - {f.name for f in fields(TrainConfig)}
    assert own <= set(values), f"missing in yaml: {own - set(values)}"
    for key, value in values.items():
        assert getattr(defaults, key) == value, key


def test_train_yaml_options_exist():
    values = yaml.safe_load((CONFIGS / "train.yaml").read_text())
    values.pop("defaults"), values.pop("hydra")
    names = {f.name for f in fields(TrainConfig)}
    assert set(values) <= names


def test_unknown_option_is_rejected():
    with pytest.raises(ValueError):
        okapi.get_algorithm("sac").Config.from_dict({"learning_rat": 1e-3})
