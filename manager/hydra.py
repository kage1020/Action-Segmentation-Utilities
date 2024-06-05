import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config(DictConfig):
    # system
    seed: int
    device: int
    verbose: bool

    # dataset
    dataset: str
    split: int
    backgrounds: list[str]
    batch_size: int
    sample_rate: int
    shuffle: bool
    base_dir: str
    split_dir: str
    gt_dir: str
    feature_dir: str
    prob_dir: str | None
    pseudo_dir: str | None

    # learning
    model_dir: str
    result_dir: str
    epochs: int
    lr: float
    weight_decay: float
    lr_gamma: float
    lr_step: int
    ensemble_weights: list[float]


def validate_config(cfg: Config):
    return cfg


class HydraManager:
    def __init__(self, config_path: str = "config"):
        self.config_path = Path(config_path)

    def __call__(self, fn):
        print(self.config_path.parent.absolute().as_posix(), self.config_path.stem)
        @hydra.main(
            config_path=self.config_path.parent.absolute().as_posix(),
            config_name=self.config_path.stem,
            version_base=None
        )
        def wrapper(cfg: Config):
            cfg = validate_config(cfg)
            return fn(cfg)
        return wrapper
