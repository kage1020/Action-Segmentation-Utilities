import hydra
from pathlib import Path

from base import Config, Base


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
            cfg = Base.validate_config(cfg)
            return fn(cfg)

        return wrapper
