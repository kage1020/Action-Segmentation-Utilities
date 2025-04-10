from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..configs.paprika import PaprikaConfig
from ..models.paprika.builder import Builder
from .main import BaseDataLoader, BaseDataset


class PaprikaNissanDataset(BaseDataset):
    def __init__(self, cfg: PaprikaConfig, train: bool = True):
        super().__init__(cfg, "PaprikaNissanDataset", train)
        self.cfg = cfg
        self.builder = Builder(cfg)

    def make_pseudo_label(self):
        self.builder.get_sim_scores()
        self.builder.get_nodes()
        self.builder.get_edges()

        self.pseudo_label_VNM = {}
        self.pseudo_label_VTM = {}
        self.pseudo_label_TCL = {}
        self.pseudo_label_NRL = {}
        if "VNM" in self.cfg.adapter_objective:
            self.pseudo_label_VNM = self.builder.get_pseudo_label_VNM()
        if "VTM" in self.cfg.adapter_objective:
            self.pseudo_label_VTM = self.builder.get_pseudo_label_VTM()
        if "TCL" in self.cfg.adapter_objective:
            self.pseudo_label_TCN = self.builder.get_pseudo_label_TCL()
        if "NRL" in self.cfg.adapter_objective:
            self.pseudo_label_NRL = self.builder.get_pseudo_label_NRL()

    def parse_pseudo_label_VNM(self, idx: int):
        pseudo_label_indices = self.pseudo_label_VNM[idx]["indices"]
        pseudo_label_VNM = np.zeros(self.cfg.dataset.num_nodes)
        for i in range(len(pseudo_label_indices)):
            pseudo_label_VNM[pseudo_label_indices[i]] = 1
        return pseudo_label_VNM

    def parse_pseudo_label_VTM(self, idx: int):
        document_task_ids = self.pseudo_label_VTM[idx]["document_tasks"]
        dataset_task_ids = self.pseudo_label_VTM[idx]["dataset_tasks"]
        pseudo_label_VTM_document = np.zeros(self.cfg.document.num_tasks)
        pseudo_label_VTM_dataset = np.zeros(self.cfg.dataset.num_tasks)
        for i in range(len(document_task_ids)):
            pseudo_label_VTM_document[document_task_ids[i]] = 1
        for i in range(len(dataset_task_ids)):
            pseudo_label_VTM_dataset[dataset_task_ids[i]] = 1
        return pseudo_label_VTM_document, pseudo_label_VTM_dataset

    def parse_pseudo_label_TCL(self, idx: int):
        document_tasknode_ids = self.pseudo_label_TCL[idx]["document_tasknodes"]
        dataset_tasknode_ids = self.pseudo_label_TCL[idx]["dataset_tasknodes"][
            "indices"
        ]
        pseudo_label_TCL_document = np.zeros(self.cfg.document.num_tasks)
        pseudo_label_TCL_dataset = np.zeros(self.cfg.dataset.num_nodes)
        for i in range(len(document_tasknode_ids)):
            pseudo_label_TCL_document[document_tasknode_ids[i]] = 1
        for i in range(len(dataset_tasknode_ids)):
            pseudo_label_TCL_dataset[dataset_tasknode_ids[i]] = 1
        return pseudo_label_TCL_document, pseudo_label_TCL_dataset

    def parse_pseudo_label_NRL(self, idx: int):
        sample = self.pseudo_label_NRL[idx]
        sub_name = [
            f"{h}-hop-{direction}"
            for direction in ["out", "in"]
            for h in range(1, self.cfg.pretrain_khop + 1)
        ]
        pseudo_label_NRL = torch.zeros((len(sub_name), self.cfg.dataset.num_nodes))
        for sub_idx in range(len(sub_name)):
            indices = sample[sub_name[sub_idx]]["indices"]
            for i in range(len(indices)):
                pseudo_label_NRL[sub_idx, indices[i]] = 1
        return pseudo_label_NRL

    def __getitem__(self, idx: int):
        video_path = Path(
            f"{self.data_dir}/{self.cfg.dataset.feature_dir}/{self.videos[idx]}.npy"
        )
        features = self.__get_video(video_path)
        gt = self.__get_gt(video_path)

        pseudo_label_VNM = self.parse_pseudo_label_VNM(idx)
        pseudo_label_VTM = self.parse_pseudo_label_VTM(idx)
        pseudo_label_TCL = self.parse_pseudo_label_TCL(idx)
        pseudo_label_NRL = self.parse_pseudo_label_NRL(idx)
        return (
            features,
            gt,
            pseudo_label_VNM,
            pseudo_label_VTM,
            pseudo_label_TCL,
            pseudo_label_NRL,
            video_path.stem,
        )


class PaprikaNissanDataLoader(BaseDataLoader, DataLoader):
    dataset: PaprikaNissanDataset

    def __init__(self, cfg: PaprikaConfig, train: bool = True, collate_fn=None):
        dataset = PaprikaNissanDataset(cfg, train)
        super(BaseDataLoader, self).__init__(
            dataset=dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=cfg.dataset.shuffle,
            collate_fn=collate_fn,
        )
