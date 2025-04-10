import hydra

from ..base import validate_config
from ..configs import Config

# FIXME: This class can't use in other files
# @hydra.main searches config files from here(asu.manager.config...).
# if we use `initialize` and `compose`, it will not create output directory and other methods also.


class HydraManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir

    def __call__(self, fn):
        @hydra.main(
            config_path=self.config_dir,
            config_name=None,
            version_base=None,
        )
        def wrapper(cfg: Config):
            is_valid = validate_config(cfg)
            if not is_valid:
                raise ValueError("Invalid configuration")

            return fn(cfg)

        return wrapper
