import ffmpeg
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from base import Base
from loader import ImageBatch


class Extractor(Base):
    def __init__(
        self,
        out_dir: str,
        video_dir: str | None = None,
        image_dir: str | None = None,
    ):
        super().__init__()
        assert (
            video_dir is not None or image_dir is not None
        ), "video_dir or image_dir must be provided"

        self.out_dir = Path(out_dir)
        video_paths = glob.glob(f"{video_dir}/*.mp4") if video_dir is not None else []
        video_paths.sort()
        self.video_paths = [Path(video_path) for video_path in video_paths]
        self.image_dir = image_dir

    def extract_features(self, model: nn.Module, model_name: str):
        image_paths = glob.glob(f"{self.image_dir}/*.png")
        image_paths += glob.glob(f"{self.image_dir}/*.jpg")
        if model_name == "i3d":
            loader = DataLoader(
                ImageBatch(image_paths, 15),
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: x[0],
                num_workers=torch.get_num_threads(),
            )
        elif model_name == "s3d":
            loader = DataLoader(
                ImageBatch(image_paths, 32),
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: x[0],
                num_workers=torch.get_num_threads(),
            )
        else:
            raise ValueError(f"model_name {model_name} is not supported")

        model.eval()
        with torch.no_grad():
            features = []
            for i, (rgb, flows) in enumerate(loader):
                rgb = rgb.to(model.device)
                flows = flows.to(model.device)
                rgb_features = model(rgb)
                flows_features = model(flows)
                features.append(
                    torch.cat([rgb_features, flows_features], dim=1).cpu().numpy()
                )

            features = np.concatenate(features, axis=0)
            np.save(self.out_dir / "features.npy", features)

    def extract_images(self):
        for video_path in self.video_paths:
            video_name = video_path.stem
            out_dir = self.out_dir / video_name
            out_dir.mkdir(parents=True, exist_ok=True)
            ffmpeg.input(str(video_path)).output(str(out_dir / "%08d.png")).run()
