import hydra
from omegaconf import DictConfig


class Config(DictConfig):
    pass


def validate_config(cfg: Config):
    return cfg


class HydraManager:
    def __init__(self, config_path: str = "config"):
        self.config_path = config_path

    def __call__(self, fn):
        @hydra.main()
        def wrapper(cfg: Config):
            cfg = validate_config(cfg)
            return fn(cfg)
        return wrapper
