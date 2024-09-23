import os
import ffmpeg
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from base import Base
from loader import ImageBatch

# TODO: support for trimmed video


class Extractor(Base):
    def __init__(
        self,
        out_dir: str,
        video_dir: str | None = None,
        image_dir: str | None = None,
    ):
        super().__init__(name="Extractor")
        assert (
            video_dir is not None or image_dir is not None
        ), "video_dir or image_dir must be provided"

        self.out_dir = Path(out_dir)
        video_paths = (
            sum([list(Path(video_dir).glob(f"**/*.{ext}")) for ext in ["mp4", "avi"]], [])
            if video_dir is not None
            else []
        )
        video_paths.sort()
        self.video_paths = [Path(video_path) for video_path in video_paths]
        self.image_dir = image_dir

        os.makedirs(self.out_dir, exist_ok=True)

    def extract_features(self, model: nn.Module, model_name: str):
        if self.image_dir is None:
            return

        image_dirs = Path(self.image_dir).glob("*")
        for image_dir in tqdm(image_dirs, leave=False):
            image_paths = sum(
                [glob.glob(f"{image_dir}/*.{ext}") for ext in ["png", "jpg", "jpeg"]],
                [],
            )
            image_paths.sort()
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
                np.save(self.out_dir / image_dir.stem / "features.npy", features)

    def extract_images(self):
        for video_path in tqdm(self.video_paths, leave=False):
            video_name = video_path.stem
            out_dir = self.out_dir / video_path.parent.name / video_name

            if out_dir.exists():
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            ffmpeg.input(str(video_path)).output(
                str(out_dir / "%08d.png"), loglevel="quiet"
            ).run()
