from dataclasses import dataclass
from einops import rearrange
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig

from base import Config
from visualizer import Visualizer
from evaluator import Evaluator
from .main import Trainer


@dataclass
class ASFormerConfig(Config):
    num_decoders: int
    num_layers: int
    r1: int
    r2: int
    num_f_maps: int
    channel_masking_rate: float
    att_type: str
    alpha: float
    p: float
    scheduler_mode: str
    scheduler_factor: float
    scheduler_patience: int
    mse_weight: float


class ASFormerCriterion(Module):
    def __init__(self, num_classes: int, mse_weight: float):
        super().__init__()
        self.ce = CrossEntropyLoss(ignore_index=-100)
        self.mse = MSELoss(reduction="none")
        self.num_classes = num_classes
        self.mse_weight = mse_weight

    def forward(self, pred: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
        pred = rearrange(pred, "b c t -> b t c")
        loss = self.ce(pred[0], labels[0])
        prev = F.log_softmax(pred[0, :-1], dim=1)
        next_ = F.log_softmax(pred.detach()[0, 1:], dim=1)
        mse = self.mse(prev, next_)
        loss += self.mse_weight * torch.mean(
            torch.clamp(mse.permute(1, 0), min=0, max=16) * mask[0, :, 1:]
        )

        return loss


class ASFormerOptimizer(Adam):
    def __init__(self, model: Module, lr: float, weight_decay: float):
        super().__init__(model.parameters(), lr=lr, weight_decay=weight_decay)


class ASFormerScheduler(ReduceLROnPlateau):
    def __init__(
        self,
        optimizer: ASFormerOptimizer,
        mode: str,
        factor: float,
        patience: int,
    ):
        super().__init__(optimizer, mode=mode, factor=factor, patience=patience)


class ASFormerTrainer(Trainer):
    def __init__(
        self,
        cfg: ASFormerConfig,
        model: Module,
    ):
        super().__init__(cfg, model, name="ASFormerTrainer")
        self.best_acc = 0
        self.best_edit = 0
        self.best_f1 = [0, 0, 0]
        self.criterion = ASFormerCriterion(cfg.dataset.num_classes, cfg.mse_weight)
        self.optimizer = ASFormerOptimizer(model, cfg.lr, cfg.weight_decay)
        self.scheduler = ASFormerScheduler(
            self.optimizer,
            mode=cfg.scheduler_mode,
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )
        self.train_evaluator = Evaluator(cfg)
        self.test_evaluator = Evaluator(cfg)
        self.visualizer = Visualizer()

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        self.model.to(self.device)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_loss: float = 0

            for features, mask, labels in tqdm(train_loader, leave=False):
                features = features.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(features, mask)
                loss: Tensor = torch.tensor(0).float().to(self.device)
                for output in outputs:
                    loss += self.criterion(output, labels, mask)
                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                pred = torch.argmax(F.softmax(outputs[-1], dim=1), dim=1)
                self.train_evaluator.add(
                    labels[0].cpu().numpy(), pred[0].cpu().detach().numpy()
                )

            if (epoch + 1) % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    f"{HydraConfig.get().runtime.output_dir}/{self.cfg.result_dir}/epoch-{epoch+1}.model",
                )

            acc, edit, f1 = self.train_evaluator.get()
            self.logger.info(
                f"Epoch {epoch+1:03d} | F1@10: {f1[0]:.3f}, F1@25: {f1[1]:.3f}, F1@50: {f1[2]:.3f}, Edit: {edit:.3f}, Acc: {acc:.3f}, Loss: {epoch_loss:.3f}"
            )
            self.visualizer.add_metrics(
                epoch,
                {
                    "Loss": epoch_loss,
                    "Acc": acc,
                    "Edit": edit,
                    "F1@10": f1[0],
                    "F1@25": f1[1],
                    "F1@50": f1[2],
                },
            )
            if self.best_f1[0] < f1[0]:
                self.best_acc = acc
                self.best_edit = edit
                self.best_f1 = f1
                torch.save(
                    self.model.state_dict(),
                    f"{HydraConfig.get().runtime.output_dir}/{self.cfg.result_dir}/best_split{self.cfg.dataset.split}.model",
                )
            self.train_evaluator.reset()
            self.scheduler.step(epoch)

            if self.cfg.val_skip:
                continue

            self.model.eval()
            epoch_loss = 0

        self.visualizer.save_metrics(
            f"{HydraConfig.get().runtime.output_dir}/{self.cfg.result_dir}"
        )

    def test(self, test_loader: DataLoader):
        with torch.no_grad():
            for features, mask, labels in test_loader:
                features = features.to(self.device)
                outputs = self.model(features, mask)
                pred = torch.argmax(F.softmax(outputs[-1], dim=1), dim=1)
                self.test_evaluator.add(labels, pred)
            acc, edit, f1 = self.test_evaluator.get()
            self.logger.info(
                f"F1@10: {f1[0]:.3f}, F1@25: {f1[1]:.3f}, F1@50: {f1[2]:.3f}, Edit: {edit:.3f}, Acc: {acc:.3f}"
            )
