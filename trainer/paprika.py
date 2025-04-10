import math

from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Module
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ..configs import PaprikaConfig, PaprikaPseudoConfig
from ..loader import BaseDataLoader
from ..trainer import NoopLoss, NoopScheduler, Trainer
from ..visualizer import Visualizer


class PaprikaCriterion(Module):
    def __init__(self, cfg: PaprikaConfig):
        super().__init__()
        self.cfg = cfg
        self.vnm = (
            BCEWithLogitsLoss(reduction="mean").to(cfg.device)
            if "VNM" in cfg.adapter_objective
            else NoopLoss().to(cfg.device)
        )
        self.vtm = [
            BCEWithLogitsLoss(reduction="mean").to(cfg.device)
            if "VTM" in cfg.adapter_objective
            else NoopLoss().to(cfg.device)
            for _ in range(2)
        ]
        self.tcl = [
            BCEWithLogitsLoss(reduction="mean").to(cfg.device)
            if "TCL" in cfg.adapter_objective
            else NoopLoss().to(cfg.device)
            for _ in range(2)
        ]
        self.nrl = [
            BCEWithLogitsLoss(reduction="mean").to(cfg.device)
            if "NRL" in cfg.adapter_objective
            else NoopLoss().to(cfg.device)
            for _ in range(2 * cfg.pretrain_khop)
        ]

    def train(self):
        self.vnm.train()
        for i in range(2):
            self.vtm[i].train()
            self.tcl[i].train()
        for i in range(2 * self.cfg.pretrain_khop):
            self.nrl[i].train()

    def forward(
        self,
        gt: tuple[Tensor, list[Tensor], list[Tensor], list[Tensor]],
        pred: tuple[Tensor, list[Tensor], list[Tensor], list[Tensor]],
    ) -> Tensor:
        gt_VNM, gt_VTM, gt_TCL, gt_NRL = gt
        pred_VNM, pred_VTM, pred_TCL, pred_NRL = pred
        vnm_loss = self.vnm(gt_VNM, pred_VNM)
        vtm_loss = sum(
            [criterion(gt_VTM[i], pred_VTM[i]) for i, criterion in enumerate(self.vtm)]
        )
        tcn_loss = sum(
            [criterion(gt_TCL[i], pred_TCL[i]) for i, criterion in enumerate(self.tcl)]
        )
        nrl_loss = sum(
            [criterion(gt_NRL[i], pred_NRL[i]) for i, criterion in enumerate(self.nrl)]
        )
        return vnm_loss + vtm_loss + tcn_loss + nrl_loss, {
            "VNM": vnm_loss,
            "VTM": vtm_loss,
            "TCN": tcn_loss,
            "NRL": nrl_loss,
        }


class PaprikaOptimizer(Adam):
    def __init__(self, cfg: PaprikaConfig, model: Module):
        params = filter(lambda p: p.requires_grad, model.parameters())
        super().__init__(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


class PaprikaScheduler:
    def __init__(
        self,
        optimizer: PaprikaOptimizer,
        cfg: PaprikaConfig,
        last_epoch: int = -1,
        training_steps: int = None,
        cycles: float = 0.5,
    ):
        if cfg.warmup_steps is None:
            self.scheduler = NoopScheduler(optimizer, last_epoch)
            return

        def lr_lambda(epoch: int) -> float:
            if cfg.warmup_steps is None:
                return cfg.lr

            assert training_steps is not None
            if epoch < cfg.warmup_steps:
                return float(epoch / max(1, cfg.warmup_steps))
            progress = float(epoch - cfg.warmup_steps) / float(
                max(1, training_steps - cfg.warmup_steps)
            )
            return max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * float(cycles) * 2.0 * progress)),
            )

        self.scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

    def step(self):
        self.scheduler.step()


class PaprikaTrainer(Trainer):
    def __init__(self, cfg: PaprikaPseudoConfig, model: Module):
        super().__init__(cfg, model, name="PaprikaTrainer")
        self.cfg = cfg
        self.criterion = PaprikaCriterion(cfg)
        self.optimizer = PaprikaOptimizer(cfg, model)
        self.visualizer = Visualizer()

    def train(self, loader: BaseDataLoader):
        self.model.to(self.device)
        self.scheduler = PaprikaScheduler(
            self.optimizer,
            self.cfg,
            last_epoch=len(loader) * self.cfg.epochs,
        )

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_loss: float = 0

            for features, pseudo_VNM, pseudo_VTM, pseudo_TCL, pseudo_NRL in tqdm(
                loader, leave=False
            ):
                features = features.to(self.device)
                pseudo_VNM = pseudo_VNM.to(self.device)
                pseudo_VTM = [vtm.to(self.device) for vtm in pseudo_VTM]
                pseudo_TCL = [tcl.to(self.device) for tcl in pseudo_TCL]
                pseudo_NRL = [nrl.to(self.device) for nrl in pseudo_NRL]
                self.optimizer.zero_grad()

                VNM_answer, VTM_answer, TCL_answer, NRL_answer = self.model(features)
                loss, dict_loss = self.criterion(
                    (pseudo_VNM, pseudo_VTM, pseudo_TCL, pseudo_NRL),
                    (VNM_answer, VTM_answer, TCL_answer, NRL_answer),
                )

                self.visualizer.add_metrics(
                    epoch,
                    {
                        "VNM Loss": dict_loss["VNM"].item(),
                        "VTM Loss": dict_loss["VTM"].item(),
                        "TCL Loss": dict_loss["TCL"].item(),
                        "NRL Loss": dict_loss["NRL"].item(),
                    },
                )

                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            self.logger.info(f"Epoch {epoch+1}, Loss: {epoch_loss}")

    def test(self, loader: BaseDataLoader):
        pass
