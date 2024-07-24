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
        self.optical_flow = cv2.optflow.DualTVL1OpticalFlow.create()
        self.to_rgb = transforms.Compose(
            [
                transforms.Resize((224, 224)),
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
        self.images = [Image.open(self.image_paths[i]) for i in indices]
        grays = [self.to_flow(im) for im in self.images]
        self.flows = [
            self.optical_flow.calc(
                np.array(grays[i], dtype=np.uint8),
                np.array(grays[i + 1], dtype=np.uint8),
                None,  # type: ignore
            )
            for i in range(len(grays) - 1)
        ]
        self.flows.append(self.flows[-1])

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
        if index >= len(self.image_paths) - self.temporal_window // 2 - 1:
            self.images.append(self.images[-1])
            self.flows.append(self.flows[-1])
        else:
            self.images.append(
                Image.open(self.image_paths[index + self.temporal_window // 2 + 1])
            )
            self.flows.append(
                self.optical_flow.calc(
                    np.array(self.to_flow(self.images[-2]), dtype=np.uint8),
                    np.array(self.to_flow(self.images[-1]), dtype=np.uint8),
                    None,  # type: ignore
                )
            )
        self.images.pop(0)
        self.flows.pop(0)
        assert (
            len(self.images) == self.temporal_window + 1
        ), "length of chunk should be the same as temporal window + 1"

        rgb = torch.stack([torch.tensor(self.to_rgb(im)) for im in self.images], dim=0)

        flows = torch.stack(
            [torch.from_numpy(flow).permute(2, 0, 1) for flow in self.flows], dim=0
        )

        rgb = rgb.permute(1, 0, 2, 3).unsqueeze(0)
        flows = flows.permute(1, 0, 2, 3).unsqueeze(0)
        return rgb, flows


class I3DDataset(Dataset):
    def __init__(self, image_dir: str, temporal_window: int):
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
    def __init__(self, video_dir: str, temporal_window: int, num_workers: int):
        dataset = I3DDataset(video_dir, temporal_window)
        super().__init__(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=lambda x: x[0],
        )
