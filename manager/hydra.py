import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
from pathlib import Path


@dataclass()
class Config(DictConfig):
    # system
    seed: int
    device: int
    verbose: bool
    val_skip: bool

    # dataset
    dataset: str
    split: int
    num_fold: int
    backgrounds: list[str]
    batch_size: int
    shuffle: bool
    base_dir: str
    split_dir: str
    gt_dir: str
    feature_dir: str
    prob_dir: str | None
    pseudo_dir: str | None
    semi_per: float

    # learning
    train: bool
    model_name: str
    model_dir: str
    result_dir: str
    epochs: int
    lr: float
    weight_decay: float
    mse_weight: float


def validate_config(cfg: Config):
    return cfg


class HydraManager:
    def __init__(self, config_path: str = "config"):
        self.config_path = Path(config_path)

    def __call__(self, fn):
        @hydra.main(
            config_path=self.config_path.parent.absolute().as_posix(),
            config_name=self.config_path.stem,
            version_base=None,
        )
        def wrapper(cfg: Config):
            cfg = validate_config(cfg)
            return fn(cfg)

        return wrapper
