from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from base.main import Base


class ImageBatch(Dataset):
    def __init__(self, image_paths: list[str] | list[Path], temporal_window: int):
        super().__init__()
        self.image_paths = image_paths
        self.temporal_window = temporal_window
        self.to_rgb = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # initialize images
        indices = [0] * (self.temporal_window // 2 + 1) + list(
            range(self.temporal_window // 2 + 1)
        )
        self.images = (
            torch.stack(
                [self.to_rgb(Image.open(self.image_paths[i])) for i in indices],
                dim=0,
            )
            if len(self.image_paths) > 0
            else torch.zeros(0)
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Parameters:
            index: index of image list
        Returns:
            rgb: rgb images, tensor of shape (1, 3, temporal_window, 224, 224)
            flows: optical flows, tensor of shape (1, 2, temporal_window, 224, 224)
        """
        # move window
        self.images = torch.roll(self.images, 1, 0)
        if index >= len(self.image_paths) - self.temporal_window // 2 - 1:
            self.images[-1] = self.images[-2]
        else:
            self.images[-1] = self.to_rgb(
                Image.open(self.image_paths[index + self.temporal_window // 2 + 1])
            )

        return self.images.permute(1, 0, 2, 3).unsqueeze(0)


class S3DDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        temporal_window: int,
        num_workers: int,
        boundary_dir: Path | None = None,
    ):
        super().__init__()
        self.image_dir = Path(image_dir)
        Base.info(f"Loading images from {self.image_dir} ...", end="")
        image_dirs = Base.get_dirs(image_dir, recursive=True)
        Base.info("Done")
        image_dirs.sort()
        self.image_dirs = image_dirs
        self.temporal_window = temporal_window
        self.num_workers = num_workers
        self.boundaries = []
        if boundary_dir is not None:
            boundaries = list(Base.get_boundaries(boundary_dir).items())
            for video_name, boundary in boundaries:
                self.boundaries.extend([(video_name, b) for b in boundary])

    def __len__(self):
        if len(self.boundaries) > 0:
            return len(self.boundaries)
        return len(self.image_dirs)

    def __getitem__(self, idx: int):
        if len(self.boundaries) > 0:
            video_name, boundaries = self.boundaries[idx]
            boundary_idx_in_video = [
                b for b in self.boundaries if b[0] == video_name
            ].index((video_name, boundaries))
            image_paths = [
                self.image_dir / video_name / f"{i:06d}.jpg"
                if (self.image_dir / video_name / f"{i:06d}.jpg").exists()
                else self.image_dir / video_name / f"{i:08d}.jpg"
                for i in range(boundaries[0] - 1, boundaries[1])
            ]
            assert len(image_paths) == boundaries[1] - boundaries[0] + 1

            return DataLoader(
                ImageBatch(image_paths, self.temporal_window),
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False,
            ), self.image_dir / (video_name + f"_{boundary_idx_in_video+1:02d}")
        else:
            image_dir = self.image_dirs[idx]
            image_paths = list(image_dir.glob("*.png"))
            image_paths += list(image_dir.glob("*.jpg"))
            image_paths.sort()

            return DataLoader(
                ImageBatch(image_paths, self.temporal_window),
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False,
            ), image_dir


class S3DDataLoader(DataLoader):
    def __init__(
        self,
        image_dir: str | Path,
        temporal_window: int,
        num_workers: int,
        boundary_dir: Path | None = None,
    ):
        dataset = S3DDataset(image_dir, temporal_window, num_workers, boundary_dir)
        super().__init__(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=lambda x: x[0],
        )
