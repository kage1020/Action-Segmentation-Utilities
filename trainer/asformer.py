import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LRScheduler
from .main import Trainer
from manager import Config


class ASFormerConfig(Config):
    sample_rate: int


class ASFormerCriterion(Module):
    def __init__(self, num_classes: int, mse_weight: float):
        self.ce = CrossEntropyLoss(ignore_index=-100)
        self.mse = MSELoss(reduction="none")
        self.num_classes = num_classes
        self.mse_weight = mse_weight

    def forward(self, pred: Tensor, labels: Tensor, mask: Tensor):
        loss = self.ce(
            pred.transpose(2, 1).contiguous().view(-1, self.num_classes),
            labels.view(-1),
        )
        prev = F.log_softmax(pred[:, :, 1:], dim=1)
        next_ = F.log_softmax(pred.detach()[:, :, :-1], dim=1)
        mse = self.mse(prev, next_)
        loss += self.mse_weight * torch.mean(
            torch.clamp(mse, min=0, max=16) * mask[:, :, 1:]
        )

        return loss


class ASFormerOptimizer(Adam):
    def __init__(self, model: Module, lr: float, weight_decay: float):
        super(ASFormerOptimizer, self).__init__(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )


class ASFormerScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, mode: str, factor: float, patience: int):
        super(ASFormerScheduler, self).__init__(
            optimizer, mode=mode, factor=factor, patience=patience, verbose=True
        )


class ASFormerTrainer(Trainer):
    def __init__(
        self,
        cfg,
        logger,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        evaluator,
    ):
        super(ASFormerTrainer, self).__init__(
            cfg, logger, model, criterion, optimizer, scheduler, evaluator
        )
        self.best_acc = 0
        self.best_edit = 0
        self.best_f1 = [0, 0, 0]

    def train(self, train_loader, test_loader):
        self.model.to(self.device)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_loss = 0

            for features, mask, labels in train_loader:
                features = features.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(features, mask)
                loss = 0
                for output in outputs:
                    loss += self.criterion(output, labels)
                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                pred = torch.argmax(F.softmax(outputs[-1], dim=1), dim=1)
                self.train_evaluator.add(
                    labels[0].cpu().numpy(), pred.cpu().detach().numpy()
                )

            if (epoch + 1) % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    f"{self.cfg.base_dir}/{self.cfg.result_dir}/epoch-{epoch+1}.model",
                )

            acc, edit, f1 = self.train_evaluator.get()
            self.logger.info(
                f"Epoch {epoch+1} | F1@10: {f1[0]:.3f}, F1@25: {f1[1]:.3f}, F1@50: {f1[2]:.3f}, Edit: {edit:.3f}, Acc: {acc:.3f}, Loss: {epoch_loss:.3f}"
            )
            if self.best_f1[0] < f1[0]:
                self.best_acc = acc
                self.best_edit = edit
                self.best_f1 = f1
                torch.save(
                    self.model.state_dict(),
                    f"{self.cfg.base_dir}/{self.cfg.result_dir}/best.model",
                )
            self.train_evaluator.reset()
            self.scheduler.step(epoch_loss)

            if self.cfg.val_skip:
                continue

            self.model.eval()
            epoch_loss = 0

            for features, mask, labels in test_loader:
                features = features.to(self.device)
                output = self.model(features, mask)
                pred = torch.argmax(F.softmax(outputs[-1], dim=1), dim=1)
                self.test_evaluator.add(labels, pred)
            acc, edit, f1 = self.test_evaluator.get()
            self.logger.info(
                f"Epoch {epoch+1} | F1@10: {f1[0]:.3f}, F1@25: {f1[1]:.3f}, F1@50: {f1[2]:.3f}, Edit: {edit:.3f}, Acc: {acc:.3f}, Loss: {epoch_loss:.3f}"
            )

    def test(self, test_loader):
        self.model.to(self.device)
        self.model.eval()
        model_path = f"{self.cfg.base_dir}/{self.cfg.model_dir}/best.model"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        with torch.no_grad():
            for features, mask, labels in test_loader:
                features = features.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features, mask)
                conf, pred = torch.max(F.softmax(outputs[-1], dim=1), dim=1)
                self.test_evaluator.add(labels, pred)
            acc, edit, f1 = self.evaluator.get()
            self.logger.info(
                f"F1@10: {f1[0]:.3f}, F1@25: {f1[1]:.3f}, F1@50: {f1[2]:.3f}, Edit: {edit:.3f}, Acc: {acc:.3f}"
            )