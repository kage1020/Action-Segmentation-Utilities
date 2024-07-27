import torch
from torch.nn import Module

from base import Base, Config


class Trainer(Base):
    def __init__(
        self,
        cfg: Config,
        model: Module,
    ):
        self.device = torch.device(
            f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"Running on {self.device}")
        self.cfg = cfg
        self.model = model

        Base.init_seed(cfg.seed)

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
