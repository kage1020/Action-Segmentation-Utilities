import math
from torch.nn import Module, BCEWithLogitsLoss
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass
from tqdm import tqdm
from base import Config

from loader import BaseDataLoader
from models.paprika.build_knowledge import obtain_external_knowledge
from trainer import Trainer, NoopLoss

from torch import Tensor


@dataclass
class PaprikaConfig(Config):
    document_dir: str
    document_num_tasks: int
    dataset_num_tasks: int

    num_nodes: int


@dataclass
class PaprikaPseudoConfig(PaprikaConfig):
    # remove_step_duplicates: bool
    # step_clustering_linkage: str
    # step_clustering_distance_thresh: float
    # step_clustering_affinity: str
    # edge_min_aggconf: int
    pass


@dataclass
class PaprikaPretrainConfig(PaprikaConfig):
    # label_find_matched_nodes_criteria: str
    # label_find_matched_nodes_for_segments_thresh: float
    # label_find_matched_nodes_for_segments_topK: int
    # label_find_tasks_criteria: str
    # label_find_tasks_thresh: float
    # label_find_tasks_topK: int
    # label_find_tasknodes_criteria: str
    # label_find_tasknodes_thresh: float
    # label_find_tasknodes_topK: int
    # label_find_neighbors_criteria: str
    # label_find_neighbors_thresh: float
    # label_find_neighbors_topK: int
    # label_khop: int

    adapter_objective: str
    pretrain_khop: int
    adapter_learning_rate: float
    adapter_weight_decay: float
    num_warmup_steps: int
    num_training_steps: int
    num_cycles: int

    s3d_hidden_dim: int
    bottleneck_dim: int
    adapter_refined_feat_dim: int
    adapter_num_classes: int | None
    # video_meta_csv_path: str
    # task_id_to_task_name_csv_path: str
    # num_partitions: int
    segment_step_sim_scores_ready: bool
    segment_step_sim_scores_DS_ready: bool
    nodes_formed: bool
    edges_formed: bool
    pseudo_label_VNM_ready: bool
    pseudo_label_VTM_ready: bool
    pseudo_label_TCL_ready: bool
    pseudo_label_NRL_ready: bool
    pseudo_label_DS_ready: bool
    partition_dataset: bool


class PaprikaCriterion(Module):
    def __init__(self, cfg: PaprikaPretrainConfig):
        self.cfg = cfg
        self.vnm = BCEWithLogitsLoss(reduction='mean').to(cfg.device) if 'VNM' in cfg.adapter_objective else NoopLoss().to(cfg.device)
        self.vtm = [BCEWithLogitsLoss(reduction='mean').to(cfg.device) if 'VTM' in cfg.adapter_objective else NoopLoss().to(cfg.device) for _ in range(2)]
        self.tcn = [BCEWithLogitsLoss(reduction='mean').to(cfg.device) if 'TCN' in cfg.adapter_objective else NoopLoss().to(cfg.device) for _ in range(2)]
        self.nrl = [BCEWithLogitsLoss(reduction='mean').to(cfg.device) if 'NRL' in cfg.adapter_objective else NoopLoss().to(cfg.device) for _ in range(2*cfg.pretrain_khop)]

    def train(self):
        self.vnm.train()
        for i in range(2):
            self.vtm[i].train()
            self.tcn[i].train()
        for i in range(2*self.cfg.pretrain_khop):
            self.nrl[i].train()

    def forward(self, VNM: Tensor, VTM: list[Tensor], TCN: list[Tensor], NRL: list[Tensor]) -> Tensor:
        vnm_loss = self.vnm(VNM)
        vtm_loss = sum([vtm_loss(VTM[i]) for i, vtm_loss in enumerate(self.vtm)])
        tcn_loss = sum([tcn_loss(TCN[i]) for i, tcn_loss in enumerate(self.tcn)])
        nrl_loss = sum([nrl_loss(NRL[i]) for i, nrl_loss in enumerate(self.nrl)])
        return vnm_loss + vtm_loss + tcn_loss + nrl_loss


class PaprikaOptimizer(Adam):
    def __init__(self, cfg: PaprikaPretrainConfig, model: Module):
        params = filter(lambda p: p.requires_grad, model.parameters())
        super().__init__(params, lr=cfg.adapter_learning_rate, weight_decay=cfg.adapter_weight_decay)


class PaprikaScheduler(LambdaLR):
    def __init__(self, optimizer: PaprikaOptimizer, cfg: PaprikaPretrainConfig, last_epoch: int = -1):
        def lr_lambda(epoch: int) -> float:
            if epoch < cfg.num_warmup_steps:
                return float(epoch / max(1, cfg.num_warmup_steps))

            progress = float(epoch - cfg.num_warmup_steps) / float(
                max(1, cfg.num_training_steps - cfg.num_warmup_steps)
            )
            return max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * float(cfg.num_cycles) * 2.0 * progress)),
            )

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class PaprikaTrainer(Trainer):
    def __init__(self, cfg: PaprikaPretrainConfig | PaprikaPseudoConfig, model: Module):
        super().__init__(cfg, model, name="PaprikaTrainer")
        self.cfg = cfg
        self.criterion = PaprikaCriterion(cfg) # type: ignore
        self.optimizer = PaprikaOptimizer(cfg, model) # type: ignore

    def make_pseudo_label(self):
        obtain_external_knowledge(self.cfg)

    def train(self, loader: BaseDataLoader):
        self.model.to(self.device)
        self.scheduler = PaprikaScheduler(self.optimizer, self.cfg, last_epoch=len(loader)*self.cfg.epochs) # type: ignore

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_loss: float = 0

            for (features, pseudo_VNM, pseudo_VTM, pseudo_TCL, pseudo_NRL) in tqdm(loader, leave=False):
                features = features.to(self.device)
                pseudo_VNM = pseudo_VNM.to(self.device)
                pseudo_VTM = [vtm.to(self.device) for vtm in pseudo_VTM]
                pseudo_TCL = [tcl.to(self.device) for tcl in pseudo_TCL]
                pseudo_NRL = [nrl.to(self.device) for nrl in pseudo_NRL]
                self.optimizer.zero_grad()

                VNM_answer, VTM_answer, TCL_answer, NRL_answer = self.model(features)
                loss = self.criterion(VNM_answer, VTM_answer, TCL_answer, NRL_answer)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        self.logger.info(f"Epoch {epoch+1}, Loss: {epoch_loss}")
