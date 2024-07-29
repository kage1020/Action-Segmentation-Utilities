from dataclasses import dataclass
from einops import rearrange
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn import Module, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader

from base import Config
from trainer import Trainer
from evaluator import Evaluator

from torch import Tensor


# TODO: modify


@dataclass
class LTContextConfig(Config):
    INPUT_DIM: int

    class MODEL:
        NUM_CLASSES: int
        MSE_LOSS_CLIP_VAL: int
        MSE_LOSS_FRACTION: float

    class LTC:
        NUM_STAGES: int
        NUM_LAYERS: int
        MODEL_DIM: int
        DIM_REDUCTION: int
        DROPOUT_PROB: float
        CONV_DILATION_FACTOR: int
        WINDOWED_ATTN_W: int
        LONG_TERM_ATTN_G: int
        USE_INSTANCE_NORM: bool
        DROPOUT_PROB: float
        CHANNEL_MASKING_PROB: float

    class ATTENTION:
        NUM_ATTN_HEADS: int
        DROPOUT: float

    class SOLVER:
        T_MAX: int
        ETA_MIN: float
        MILESTONE: int


class LTContextCriterion(Module):
    def __init__(self, num_classes: int, mse_clip_value: int, mse_weight: float):
        self.ce = CrossEntropyLoss(ignore_index=-100)
        self.mse = MSELoss(reduction="none")
        self.num_classes = num_classes
        self.mse_clip_value = mse_clip_value
        self.mse_weight = mse_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits (Tensor): [n_stages, batch_size, n_classes, n_frames]
            labels (Tensor): [batch_size, n_frames]
        """
        loss = {"loss_ce": torch.tensor(0).float().to(logits.device), "loss_mse": torch.tensor(0).float().to(logits.device)}
        targets = rearrange(targets, "b f -> (b f)")
        for p in logits:
            loss["loss_ce"] += self.ce(rearrange(p, "b c f -> (b f) c"), targets)
            loss["loss_mse"] += torch.mean(
                torch.clamp(
                    self.mse(
                        F.log_softmax(p[:, :, 1:], dim=1),
                        F.log_softmax(p.detach()[:, :, :-1], dim=1),
                    ),
                    min=0,
                    max=self.mse_clip_value,
                )
            )

        return loss["loss_ce"] + loss["loss_mse"] * self.mse_weight


class LTContextOptimizer(Adam):
    def __init__(
        self, model: Module, lr: float, betas: tuple[float, float], weight_decay: float
    ):
        params = filter(lambda p: p.requires_grad, model.parameters())
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)


class LTContextScheduler(SequentialLR):
    def __init__(
        self, optimizer: LTContextOptimizer, T_max: int, eta_min: float, milestone: int
    ):
        identity = LinearLR(optimizer, start_factor=1.0)
        main_lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=T_max - milestone, eta_min=eta_min
        )
        super().__init__(optimizer, [identity, main_lr_scheduler], [milestone])


class LTContextTrainer(Trainer):
    def __init__(self, cfg: LTContextConfig, model: Module):
        super().__init__(cfg, model)
        self.criterion = LTContextCriterion(
            cfg.MODEL.NUM_CLASSES,
            cfg.MODEL.MSE_LOSS_CLIP_VAL,
            cfg.MODEL.MSE_LOSS_FRACTION,
        )
        self.optimizer = LTContextOptimizer(
            model, cfg.lr, (0.9, 0.999), cfg.weight_decay
        )
        self.scheduler = LTContextScheduler(
            self.optimizer, cfg.SOLVER.T_MAX, cfg.SOLVER.ETA_MIN, cfg.SOLVER.MILESTONE
        )
        self.train_evaluator = Evaluator(cfg)
        self.test_evaluator = Evaluator(cfg)
        self.best_acc = 0
        self.best_edit = 0
        self.best_f1 = [0, 0, 0]

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        for epoch in range(self.cfg.epochs):
            self.model.to(self.device)
            for features, target, masks in train_loader:
                features = features.to(self.device)
                target = target.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(features, masks)
                loss = self.criterion(logits, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
