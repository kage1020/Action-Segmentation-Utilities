import hydra

from base import Config, Base


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
            is_valid = Base.validate_config(cfg)
            if not is_valid:
                raise ValueError("Invalid configuration")

            return fn(cfg)

        return wrapper
