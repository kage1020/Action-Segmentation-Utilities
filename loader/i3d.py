import glob
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImageBatch(Dataset):
    def __init__(self, image_paths: list[str] | list[Path], temporal_window: int):
        super(ImageBatch, self).__init__()
        self.image_paths = image_paths
        self.temporal_window = temporal_window
        self.optical_flow = cv2.optflow.DualTVL1OpticalFlow.create() # type: ignore
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
        images = [Image.open(self.image_paths[i]) for i in indices]
        grays = [self.to_flow(im) for im in images.copy()]
        flows = [
            self.optical_flow.calc(
                np.array(grays[i], dtype=np.uint8),
                np.array(grays[i + 1], dtype=np.uint8),
                None,
            )
            for i in range(len(grays) - 1)
        ]
        flows.append(flows[-1])
        self.images = torch.stack([self.to_rgb(im) for im in images], dim=0) # type: ignore
        self.flows = torch.stack([torch.from_numpy(flow) for flow in flows], dim=0)

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
        if index >= len(self.image_paths) - self.temporal_window // 2 - 1:
            self.images[-1] = self.images[-2]
            self.flows[-1] = self.flows[-2]
        else:
            self.images[-1] = self.to_rgb( # type: ignore
                Image.open(self.image_paths[index + self.temporal_window // 2 + 1])
            )
            self.flows[-1] = torch.from_numpy(self.optical_flow.calc(
                np.array(self.to_flow(self.images[-2]), dtype=np.uint8),
                np.array(self.to_flow(self.images[-1]), dtype=np.uint8),
                None,
            ))
        assert (
            len(self.images) == self.temporal_window + 1
        ), "length of chunk should be the same as temporal window + 1"

        rgb = self.images.permute(1, 0, 2, 3).unsqueeze(0)
        flows = self.flows.permute(1, 0, 2, 3).unsqueeze(0)
        return rgb, flows


class I3DDataset(Dataset):
    def __init__(self, image_dir: str | Path, temporal_window: int):
        super().__init__()
        image_paths = glob.glob(f"{image_dir}/*.png")
        image_paths += glob.glob(f"{image_dir}/*.jpg")
        self.image_paths = [Path(image_path) for image_path in image_paths]
        self.temporal_window = temporal_window

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        video_path = self.image_paths[idx]
        video_name = video_path.stem
        out_dir = video_path.parent / video_name
        image_paths = list(out_dir.glob("*.png"))
        image_paths += list(out_dir.glob("*.jpg"))
        image_paths.sort()

        return ImageBatch(image_paths, self.temporal_window)


class I3DDataLoader(DataLoader):
    def __init__(self, image_dir: str | Path, temporal_window: int, num_workers: int):
        dataset = I3DDataset(image_dir, temporal_window)
        super().__init__(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=lambda x: x[0],
        )
