import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from base import Base
from configs import Config


class NoopDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, idx: int):
        return 0


class BaseDataset(Dataset, Base):
    def __init__(self, cfg: Config, name: str = "BaseDataset", train: bool = True):
        super(Dataset, self).__init__(
            name=name,
            mapping_path=f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.mapping_path}",
            actions_path=f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.actions_path}",
            matching_path=f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.matching_path}",
            has_mapping_header=cfg.dataset.has_mapping_header,
            mapping_separator=cfg.dataset.mapping_separator,
            has_actions_header=cfg.dataset.has_actions_header,
            actions_action_separator=cfg.dataset.actions_action_separator,
            actions_class_separator=cfg.dataset.actions_class_separator,
            matching_separator=cfg.dataset.matching_separator,
        )
        self.cfg = cfg
        self.data_dir = f"{cfg.dataset.base_dir}/{cfg.dataset.name}"
        self.phase = "train" if train else "test"

        self._load_videos()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx: int):
        video_path = Path(
            f"{self.data_dir}/{self.cfg.dataset.feature_dir}/{self.videos[idx]}.npy"
        )
        features = self._get_video(video_path)
        mask = torch.ones(len(self.text_to_int), features.size(1)).float()
        gt = self._get_gt(video_path)
        return features, mask, gt, video_path.stem

    def _load_videos(self):
        with open(
            f"{self.data_dir}/{self.cfg.dataset.split_dir}/{self.phase}.split{self.cfg.dataset.split}.{self.cfg.dataset.semi_per:.2f}.bundle",
            "r",
        ) as f:
            lines = f.readlines()
            self.videos = [Path(line.strip()).stem for line in lines if line.strip()]
        self.logger.info(f"{len(self.videos)} videos loaded")

    def _get_video(self, video_path: Path):
        features = np.load(video_path)
        features = torch.from_numpy(features).float()
        features = features[:, :: self.cfg.dataset.sampling_rate]
        return features

    def _get_gt(self, video_path: Path):
        gt_path = Path(
            f"{self.data_dir}/{self.cfg.dataset.gt_dir}/{video_path.stem}.txt"
        )
        if gt_path.exists():
            with open(gt_path, "r") as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
            labels = [self.text_to_int[label] for label in labels]
            labels = torch.tensor(labels).long()[:: self.cfg.dataset.sampling_rate]
            return labels
        else:
            return torch.tensor([])

    def _get_pseudo(self, video_path: Path):
        pseudo_path = Path(
            f"{self.data_dir}/{self.cfg.dataset.pseudo_dir}/{video_path.stem}.txt"
        )
        if pseudo_path.exists():
            with open(pseudo_path, "r") as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
            labels = [self.text_to_int[label] for label in labels]
            labels = torch.tensor(labels).long()[:: self.cfg.dataset.sampling_rate]
            return labels
        else:
            return torch.tensor([])

    def _get_prob(self, video_path: Path):
        prob_path = Path(
            f"{self.data_dir}/{self.cfg.dataset.prob_dir}/{video_path.stem}.npy"
        )
        if prob_path.exists():
            probabilities = torch.from_numpy(np.load(prob_path)).float()
            return probabilities
        else:
            return torch.tensor([])

    def generate_pseudo_labels(self, model: Module):
        os.makedirs(f"{self.data_dir}/{self.cfg.dataset.pseudo_dir}", exist_ok=True)
        os.makedirs(f"{self.data_dir}/{self.cfg.dataset.prob_dir}", exist_ok=True)

        model.eval()
        with torch.no_grad():
            for video in self.videos:
                video_path = Path(
                    f"{self.data_dir}/{self.cfg.dataset.feature_dir}/{video}.npy"
                )
                gt_path = Path(
                    f"{self.data_dir}/{self.cfg.dataset.gt_dir}/{video_path.stem}.txt"
                )
                if gt_path.exists():
                    continue

                features = self._get_video(video_path)
                features = features.unsqueeze(0)
                features = features.to(model.device)
                output = model(features)
                pseudo_probs = F.softmax(output, dim=1)
                pseudo_labels = torch.argmax(pseudo_probs, dim=1)

                pseudo_path = Path(
                    f"{self.data_dir}/{self.cfg.dataset.pseudo_dir}/{video_path.stem}.txt"
                )
                prob_path = Path(
                    f"{self.data_dir}/{self.cfg.dataset.prob_dir}/{video_path.stem}.npy"
                )
                with open(pseudo_path, "w") as f:
                    for label in pseudo_labels:
                        f.write(f"{self.int_to_text[int(label)]}\n")
                np.save(prob_path, pseudo_probs.cpu().numpy())


class BaseDataLoader(DataLoader):
    dataset: BaseDataset

    def __init__(
        self, dataset: BaseDataset, cfg: Config, train: bool = True, collate_fn=None
    ):
        self.dataset = dataset
        super().__init__(
            dataset=dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=cfg.dataset.shuffle,
            collate_fn=collate_fn,
        )
