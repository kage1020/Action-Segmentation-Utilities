import os
import torch
from torch.nn import Module
from hydra.core.hydra_config import HydraConfig

from asu.base import Base, Config


class Trainer(Base):
    def __init__(
        self,
        cfg: Config,
        model: Module,
        name: str = "Trainer",
    ):
        super().__init__(name=name)
        self.device = torch.device(
            f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"Running on {self.device}")
        self.cfg = cfg
        self.model = model
        self.hydra_dir = HydraConfig.get().runtime.output_dir

        os.makedirs(f"{self.hydra_dir}/{cfg.result_dir}", exist_ok=True)
        Base.init_seed(cfg.seed)

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
