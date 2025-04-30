import os
from pathlib import Path

import ffmpeg
import numpy as np
import torch
from tqdm import tqdm

from ..base import Base, get_device
from ..loader.i3d import I3DDataLoader
from ..loader.s3d import S3DDataLoader
from .models.i3d import I3D
from .models.s3d_howto100m import S3D

# TODO: support for trimmed video


class Extractor(Base):
    def __init__(
        self,
        out_dir: str,
        video_dir: str | None = None,
        image_dir: str | None = None,
        flow_dir: str | None = None,
        boundary_dir: str | None = None,
    ):
        super().__init__(name="Extractor")
        assert video_dir is not None or image_dir is not None, (
            "video_dir or image_dir must be provided"
        )

        self.out_dir = Path(out_dir)
        self.video_paths = (
            sum(
                [list(Path(video_dir).glob(f"**/*.{ext}")) for ext in ["mp4", "avi"]],
                [],
            )
            if video_dir is not None
            else []
        )
        self.video_paths.sort()
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.flow_dir = Path(flow_dir) if flow_dir is not None else None
        self.boundary_dir = Path(boundary_dir) if boundary_dir is not None else None

        os.makedirs(self.out_dir, exist_ok=True)

    def extract_i3d_features(self, rgb_model: I3D, flow_model: I3D):
        if self.image_dir is None:
            return

        loader = I3DDataLoader(
            image_dir=self.image_dir,
            flow_dir=self.flow_dir,
            temporal_window=15,
            num_workers=1,
            boundary_dir=self.boundary_dir,
            logger=self.logger,
        )
        with torch.no_grad():
            for batch, image_dir in tqdm(loader, leave=False):
                out_path = self.out_dir / f"{image_dir.relative_to(self.image_dir)}.npy"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if len(batch) == 0 or out_path.exists():
                    continue
                features = []
                for rgb, flows in tqdm(batch, leave=False):
                    rgb = rgb.to(get_device())
                    flows = flows.to(get_device())
                    rgb_features = rgb_model.extract_features(rgb[0])
                    flows_features = flow_model.extract_features(flows[0])
                    features.append(
                        torch.cat([rgb_features.squeeze(), flows_features.squeeze()])
                        .cpu()
                        .numpy()
                    )

                features = np.stack(features)
                np.save(out_path, features)

    def extract_s3d_features(self, model: S3D):
        if self.image_dir is None:
            return

        model.eval()
        loader = S3DDataLoader(
            image_dir=self.image_dir,
            temporal_window=31,
            num_workers=1,
            boundary_dir=self.boundary_dir,
        )
        with torch.no_grad():
            for rgb, image_dir in tqdm(loader, leave=False):
                out_path = self.out_dir / f"{image_dir.relative_to(self.image_dir)}.npy"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if len(rgb) == 0 or out_path.exists():
                    continue
                features = []
                for batch in tqdm(rgb, leave=False):
                    batch = batch.to(get_device())
                    rgb_features = model(batch[0])["video_embedding"]
                    features.append(rgb_features.cpu().numpy())

                features = np.concatenate(features, axis=0)
                np.save(out_path, features)

    def extract_images(self):
        if self.image_dir is None:
            return

        for video_path in tqdm(self.video_paths, leave=False, desc="Extracting images"):
            out_dir = self.image_dir / video_path.relative_to(
                video_path.parent
            ).with_suffix("")

            if out_dir.exists():
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            ffmpeg.input(str(video_path)).output(
                str(out_dir / "%08d.jpg"), loglevel="quiet"
            ).run()
