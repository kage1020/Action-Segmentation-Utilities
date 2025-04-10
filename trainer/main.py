import os

import torch
from hydra.core.hydra_config import HydraConfig
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler

from ..base import Base, init_seed
from ..configs import Config


class NoopLoss(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, device=x.device)


class NoopScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        pass


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
        init_seed(cfg.seed)

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
