from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from asu.base.main import Config
from .main import BaseDataset


class C2FTCNBreakfastDataset(BaseDataset):
    zoom_crop = (0.5, 2)
    smallest_cut = 1.0
    unlabeled = []

    def __init__(self, cfg: Config, train: bool = True, unsupervised: bool = False):
        super(C2FTCNBreakfastDataset, self).__init__(
            cfg, "C2FTCNBreakfastDataset", train
        )

        data = []
        for video in self.videos:
            video_id = video.split(".txt")[0]
            video_path = Path(
                f"{self.cfg.dataset.base_dir}{self.cfg.dataset}/{self.cfg.dataset.feature_dir}/{video_id}.npy"
            )
            if not unsupervised:
                gt_path = Path(
                    f"{self.cfg.dataset.base_dir}{self.cfg.dataset}/{self.cfg.dataset.gt_dir}/{video_id}.txt"
                )
            else:
                gt_path = Path(
                    f"{self.cfg.dataset.base_dir}{self.cfg.dataset}/{self.cfg.dataset.pseudo_dir}/{video_id}.txt"
                )

            if gt_path.exists():
                with open(gt_path, "r") as f:
                    gt = f.readlines()
                    gt = [line.strip() for line in gt if line.strip() != ""]
                    gt = [self.text_to_int[line] for line in gt]
            else:
                gt = []
                self.unlabeled.append(video_path)
                data.append((video_path, np.array(gt, dtype=int), 0, 0))
                continue

            num_frames = len(gt)
            start_frames = []
            end_frames = []
            for start in range(
                0,
                num_frames,
                self.cfg.dataset.max_frames_per_video * self.cfg.chunk_size,
            ):
                start_frames.append(start)
                max_end = start + (self.cfg.max_frames_per_video * self.cfg.chunk_size)
                end_frames.append(max_end if max_end < num_frames else num_frames)

            for start, end in zip(start_frames, end_frames):
                data.append((video_path, np.array(gt, dtype=int), start, end))

        self.videos = data

    def __getitem__(self, index):
        video_path, gt, start, end = self.videos[index]
        features = np.load(video_path)
        end = min(end, start + (self.cfg.max_frames_per_video * self.cfg.chunk_size))
        num_frames = end - start

        chunks = torch.zeros((self.cfg.max_frames_per_video, self.cfg.feature_size))
        labels = torch.ones(self.cfg.max_frames_per_video, dtype=torch.long) * -100

        do_augmentation = (not self.phase) and np.random.randint(0, 2) == 0
        if do_augmentation:
            aug_start = np.random.uniform(low=0.0, high=1.0 - self.smallest_cut)
            aug_len = np.random.uniform(low=self.smallest_cut, high=1.0 - aug_start)
            aug_end = aug_start + aug_len
            max_chunk_size = int(self.cfg.chunk_size / self.zoom_crop[0])
            min_chunk_size = max(
                np.ceil(num_frames / self.cfg.max_frames_per_video),
                int(self.cfg.chunk_size / self.zoom_crop[1]),
            )
            aug_chunk_size = int(
                np.exp(
                    np.random.uniform(
                        low=np.log(min_chunk_size), high=np.log(max_chunk_size)
                    )
                )
            )
            num_aug_frames = np.ceil(int(aug_len * num_frames) / aug_chunk_size)
            if num_aug_frames > self.cfg.max_frames_per_video:
                num_aug_frames = self.cfg.max_frames_per_video
                aug_chunk_size = int(np.ceil(aug_len * num_frames / num_aug_frames))

            aug_offset = 0
            aug_start_frame = start + int(aug_start * num_frames)
            aug_end_frame = start + int(aug_end * num_frames)
        else:
            aug_offset = 0
            aug_start_frame = start
            aug_end_frame = end
            aug_chunk_size = self.cfg.chunk_size

        if num_frames == 0:
            return chunks, 0, labels, video_path.stem

        count = 0
        for i in range(aug_start_frame, aug_end_frame, aug_chunk_size):
            end_frame = min(aug_end_frame, i + aug_chunk_size)
            chunk = features[:, i:end_frame]
            values, counts = np.unique(gt[i:end_frame], return_counts=True)
            labels[count] = values[np.argmax(counts)]
            chunks[aug_offset + count, :] = torch.tensor(
                np.max(chunk, axis=-1), dtype=torch.float32
            )
            count += 1

        return chunks, count, labels, video_path.stem


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    chunks, counts, labels, video_ids = zip(*batch)
    return torch.stack(chunks), torch.stack(counts), torch.stack(labels), video_ids


class C2FTCNBreakfastDataLoader(DataLoader):
    def __init__(self, cfg: Config, train: bool = True):
        dataset = C2FTCNBreakfastDataset(cfg, train)

        def init_fn(worker_id):
            np.random.seed(cfg.seed)

        super(C2FTCNBreakfastDataLoader, self).__init__(
            dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=train,
            pin_memory=False,
            num_workers=0,
            collate_fn=collate_fn,
            worker_init_fn=init_fn,
        )
