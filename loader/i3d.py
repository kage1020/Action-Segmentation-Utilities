from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..base import Base, get_boundaries, get_dirs
from ..logger import Logger, log


class ImageBatch(Dataset):
    def __init__(
        self,
        image_paths: list[str] | list[Path],
        flow_paths: list[str] | list[Path],
        temporal_window: int,
    ):
        super(ImageBatch, self).__init__()
        self.image_paths = image_paths
        self.flow_paths = flow_paths
        self.temporal_window = temporal_window
        self.to_rgb = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.to_flow = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(),
            ]
        )

        # initialize rgb images and flows
        indices = [0] * (self.temporal_window // 2 + 1) + list(
            range(self.temporal_window // 2 + 1)
        )
        pre_images = torch.empty((self.temporal_window + 1, 3, 224, 224))
        pre_flows = torch.empty((self.temporal_window + 1, 2, 224, 224))
        for i in range(len(indices)):
            img = Image.open(self.image_paths[indices[i]])
            pre_images[i] = self.to_rgb(img)

            if i < len(indices) - 1:
                flow = np.load(self.flow_paths[indices[i]])  # shape: (H, W, 2)
                flow_tensor = torch.from_numpy(flow).permute(2, 0, 1)  # (2, H, W)
                flow_tensor = transforms.functional.resize(flow_tensor, [224, 224])
                pre_flows[i] = flow_tensor
            else:
                pre_flows[i] = pre_flows[i - 1]
        self.images = pre_images
        self.flows = pre_flows

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Parameters:
            index: index of image list
        Returns:
            rgb: rgb images, tensor of shape (1, temporal_window, 3, 224, 224)
            flows: optical flows, tensor of shape (1, temporal_window, 2, 224, 224)
        """
        # move window
        self.images = torch.roll(self.images, 1, 0)
        self.flows = torch.roll(self.flows, 1, 0)

        next_index = index + self.temporal_window // 2 + 1
        if next_index >= len(self.image_paths):
            self.images[-1] = self.images[-2]
            self.flows[-1] = self.flows[-2]
        else:
            img = Image.open(self.image_paths[next_index])
            self.images[-1] = self.to_rgb(img)

            flow = np.load(self.flow_paths[next_index - 1])
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1)
            flow_tensor = transforms.functional.resize(flow_tensor, [224, 224])
            self.flows[-1] = flow_tensor

        assert len(self.images) == self.temporal_window + 1, (
            "length of chunk should be the same as temporal window + 1"
        )

        rgb = self.images.permute(1, 0, 2, 3).unsqueeze(0)
        flows = self.flows.permute(1, 0, 2, 3).unsqueeze(0)
        return rgb, flows


class I3DDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        flow_dir: str | Path,
        temporal_window: int,
        num_workers: int,
        boundary_dir: Path | None = None,
        logger: Logger = Base,
    ):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.flow_dir = Path(flow_dir)
        logger.info(f"Loading images from {self.image_dir} ...", end="")
        image_dirs = get_dirs(image_dir, recursive=True)
        log("Done")
        image_dirs.sort()
        self.image_dirs = image_dirs
        self.temporal_window = temporal_window
        self.num_workers = num_workers
        self.boundary_list: list[tuple[str, tuple[int, int]]] = []
        self.boundary_dict: dict[str, list[tuple[int, int]]] = {}
        if boundary_dir is not None:
            boundary_list = list(get_boundaries(boundary_dir).items())
            for video_name, boundaries in boundary_list:
                self.boundary_list.extend([(video_name, b) for b in boundaries])
                self.boundary_dict[video_name] = boundaries

    def __len__(self):
        if len(self.boundary_list) > 0:
            return len(self.boundary_list)
        return len(self.image_dirs)

    def __getitem__(self, idx: int):
        if len(self.boundary_list) > 0:
            video_name, boundary = self.boundary_list[idx]
            boundary_idx_in_video = self.boundary_dict[video_name].index(boundary)
            existing_images = set(
                f.name for f in (self.image_dir / video_name).glob("*.jpg")
            )
            image_paths = [
                self.image_dir / video_name / f"{i:06d}.jpg"
                if f"{i:06d}.jpg" in existing_images
                else self.image_dir / video_name / f"{i:08d}.jpg"
                for i in range(boundary[0], boundary[1] + 1)
            ]
            flow_paths = [
                self.flow_dir / video_name / f"{i:06d}.npy"
                for i in range(boundary[0], boundary[1] + 1)
            ]
            assert len(image_paths) == boundary[1] - boundary[0] + 1

            video_path = self.image_dir / (
                video_name + f"_{boundary_idx_in_video + 1:02d}"
            )
            loader = DataLoader(
                ImageBatch(image_paths, flow_paths, self.temporal_window),
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False,
            )
            return loader, video_path
        else:
            image_dir = self.image_dirs[idx]
            image_paths = list(image_dir.glob("*.png"))
            image_paths += list(image_dir.glob("*.jpg"))
            image_paths.sort()
            flow_paths = [
                self.flow_dir
                / image_dir.relative_to(self.image_dir).with_suffix(".npy")
                for _ in range(len(image_paths))
            ]
            assert len(image_paths) == len(flow_paths)

            return DataLoader(
                ImageBatch(image_paths, flow_paths, self.temporal_window),
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False,
            ), image_dir


class I3DDataLoader(DataLoader):
    def __init__(
        self,
        image_dir: str | Path,
        flow_dir: str | Path,
        temporal_window: int,
        num_workers: int,
        boundary_dir: Path | None = None,
        logger: Logger | None = Base,
    ):
        dataset = I3DDataset(
            image_dir,
            flow_dir,
            temporal_window,
            num_workers,
            boundary_dir,
            logger=logger,
        )
        super().__init__(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=lambda x: x[0],
        )
