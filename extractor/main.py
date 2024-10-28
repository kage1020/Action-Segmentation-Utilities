import os
import ffmpeg
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from base import Base
from loader import I3DDataLoader, S3DDataLoader

# TODO: support for trimmed video


class Extractor(Base):
    def __init__(
        self,
        out_dir: str,
        video_dir: str | None = None,
        image_dir: str | None = None,
        boundary_dir: str | None = None,
    ):
        super().__init__(name="Extractor")
        assert (
            video_dir is not None or image_dir is not None
        ), "video_dir or image_dir must be provided"

        self.out_dir = Path(out_dir)
        video_paths = (
            sum(
                [list(Path(video_dir).glob(f"**/*.{ext}")) for ext in ["mp4", "avi"]],
                [],
            )
            if video_dir is not None
            else []
        )
        video_paths.sort()
        self.video_paths = [Path(video_path) for video_path in video_paths]
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.boundary_dir = Path(boundary_dir) if boundary_dir is not None else None

        os.makedirs(self.out_dir, exist_ok=True)

    def extract_features(self, model: nn.Module, model_name: str):
        if self.image_dir is None:
            return

        model.eval()
        if model_name == "i3d":
            loader = I3DDataLoader(
                image_dir=self.image_dir,
                temporal_window=15,
                num_workers=1,
            )
            with torch.no_grad():
                features = []
                for rgb, flows in loader:
                    rgb = rgb.to(Base.get_device())
                    flows = flows.to(Base.get_device())
                    rgb_features = model(rgb)
                    flows_features = model(flows)
                    features.append(
                        torch.cat([rgb_features, flows_features], dim=1).cpu().numpy()
                    )

                features = np.concatenate(features, axis=0)
                np.save(self.out_dir / self.image_dir.stem / "features.npy", features)
        elif model_name == "s3d":
            loader = S3DDataLoader(
                image_dir=self.image_dir,
                temporal_window=31,
                num_workers=1,
                boundary_dir=self.boundary_dir,
            )
            with torch.no_grad():
                for rgb, image_dir in tqdm(loader, leave=False):
                    out_path = (
                        self.out_dir
                        / image_dir.relative_to(self.image_dir)
                        / "features.npy"
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if len(rgb) == 0 or out_path.exists():
                        continue
                    features = []
                    for batch in tqdm(rgb, leave=False):
                        batch = batch.to(Base.get_device())
                        rgb_features = model(batch[0])["video_embedding"]
                        features.append(rgb_features.cpu().numpy())

                    features = np.concatenate(features, axis=0)
                    np.save(out_path, features)
        else:
            raise ValueError(f"model_name {model_name} is not supported")

    def extract_images(self):
        if self.image_dir is None:
            return

        for video_path in tqdm(self.video_paths, leave=False):
            video_name = video_path.stem
            out_dir = self.out_dir / video_path.parent.name / video_name

            if out_dir.exists():
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            ffmpeg.input(str(video_path)).output(
                str(out_dir / "%08d.png"), loglevel="quiet"
            ).run()
