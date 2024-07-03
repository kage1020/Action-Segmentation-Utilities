import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from copy import deepcopy
from manager import Config
from utils import init_seed
from logger import Logger
from evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        cfg: Config,
        logger: Logger,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        evaluator: Evaluator,
    ):
        self.device = torch.device(
            f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Running on: {self.device}")
        self.cfg = cfg
        self.logger = logger
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_evaluator = deepcopy(evaluator)
        self.test_evaluator = deepcopy(evaluator)
        self.best_score = 0

        init_seed(cfg.seed)

    def train(self, train_loader, test_loader):
        self.model.to(self.device)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_loss = 0

            for data, target in train_loader:
                data = data.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step(epoch_loss)

            self.model.eval()
            for data, target in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                self.test_evaluator.add(target, data)
            score = self.test_evaluator.get()
            self.logger.info(f"epoch {epoch} score: {score}")

            torch.save(
                self.model.state_dict(),
                f"{self.cfg.base_dir}/{self.cfg.result_dir}/epoch-{epoch+1}.model",
            )
            if score > self.best_score:
                self.best_score = score
                torch.save(
                    self.model.state_dict(),
                    f"{self.cfg.base_dir}/{self.cfg.result_dir}/best.model",
                )

    def test(self, test_loader):
        self.model.to(self.device)

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                self.test_evaluator.add(target, output)
            score = self.test_evaluator.get()
            self.logger.info(f"test score: {score}")
