from dataclasses import asdict, dataclass
from einops import rearrange
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn import Module, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig

from base import Config
from trainer import Trainer
from evaluator import Evaluator

from torch import Tensor


@dataclass
class LTCConfig:
    num_stages: int
    num_layers: int
    model_dim: int
    dim_reduction: int
    dropout_prob: float
    conv_dilation_factor: int
    windowed_attn_w: int
    long_term_attn_g: int
    use_instance_norm: bool
    channel_masking_prob: float


@dataclass
class AttentionConfig:
    num_attn_heads: int
    dropout: float


@dataclass
class SolverConfig:
    t_max: int
    eta_min: float
    milestone: int


@dataclass
class LTContextConfig(Config):
    mse_clip_val: float
    mse_weight: float
    LTC: LTCConfig
    ATTENTION: AttentionConfig
    SOLVER: SolverConfig

    def __post__init__(self):
        super().__post_init__()
        self.LTC = LTCConfig(**asdict(self.LTC))
        self.ATTENTION = AttentionConfig(**asdict(self.ATTENTION))
        self.SOLVER = SolverConfig(**asdict(self.SOLVER))


class LTContextCriterion(Module):
    def __init__(self, num_classes: int, mse_clip_value: float, mse_weight: float):
        super().__init__()
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
        loss = {
            "loss_ce": torch.tensor(0).float().to(logits.device),
            "loss_mse": torch.tensor(0).float().to(logits.device),
        }
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


def collate_fn(batch):
    features, _, target = zip(*batch)
    features = [rearrange(f, "c f -> f c") for f in features]
    features = pad_sequence(features, batch_first=True)
    features = rearrange(features, "b f c -> b c f")
    target = pad_sequence(target, batch_first=True, padding_value=-100)
    masks = torch.where(target == -100, 0, 1)
    masks = masks[:, None, :].bool()
    return features, masks, target


class LTContextTrainer(Trainer):
    def __init__(self, cfg: LTContextConfig, model: Module):
        super().__init__(cfg, model, name="LTContextTrainer")
        self.best_acc = 0
        self.best_edit = 0
        self.best_f1 = [0, 0, 0]
        self.criterion = LTContextCriterion(
            cfg.dataset.num_classes,
            cfg.mse_clip_val,
            cfg.mse_weight,
        )
        self.optimizer = LTContextOptimizer(
            model, cfg.lr, (0.9, 0.999), cfg.weight_decay
        )
        self.scheduler = LTContextScheduler(
            self.optimizer, cfg.SOLVER.t_max, cfg.SOLVER.eta_min, cfg.SOLVER.milestone
        )
        self.train_evaluator = Evaluator(cfg)
        self.test_evaluator = Evaluator(cfg)

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        self.model.to(self.device)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_loss: float = 0

            for features, masks, targets in tqdm(train_loader, leave=False):
                features = features.to(self.device)
                masks = masks.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()

                logits = self.model(features, masks)
                loss = self.criterion(logits, targets)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                for logit, target in zip(logits, targets):
                    pred = torch.argmax(F.softmax(logit, dim=1), dim=1)[0]
                    self.train_evaluator.add(
                        target.cpu().numpy(), pred.cpu().detach().numpy()
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
            if self.best_f1[0] < f1[0]:
                self.best_acc = acc
                self.best_edit = edit
                self.best_f1 = f1
                torch.save(
                    self.model.state_dict(),
                    f"{HydraConfig.get().runtime.output_dir}/{self.cfg.result_dir}/best_split{self.cfg.dataset.split}.model",
                )
            self.train_evaluator.reset()
            self.scheduler.step()

            if not self.cfg.val_skip:
                self.test(test_loader)

    def test(self, test_loader: DataLoader):
        self.model.to(self.device)
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for features, masks, target in tqdm(test_loader):
                features = features.to(self.device)
                masks = masks.to(self.device)
                target = target.to(self.device)

                logits = self.model(features, masks)
                loss = self.criterion(logits, target)
                epoch_loss += loss.item()

                pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
                self.test_evaluator.add(
                    target.cpu().numpy(), pred.cpu().detach().numpy()
                )

            acc, edit, f1 = self.test_evaluator.get()
            self.logger.info(
                f"Test | F1@10: {f1[0]:.3f}, F1@25: {f1[1]:.3f}, F1@50: {f1[2]:.3f}, Edit: {edit:.3f}, Acc: {acc:.3f}, Loss: {epoch_loss:.3f}"
            )
