import random
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
from trainer import PaprikaConfig
from .helper import get_all_video_ids


def get_samples(cfg: PaprikaConfig):
    videos = get_all_video_ids(cfg)

    samples = list()
    for v in tqdm(videos):
        video_s3d = np.load(
            Path(cfg.dataset.base_dir) / cfg.dataset.name / "features" / f"{v}.npy"
        )

        for c_idx in range(video_s3d.shape[0]):
            samples.append((v, c_idx))
    random.shuffle(samples)

    samples_id2name = dict()
    samples_name2id = dict()
    for i in range(len(samples)):
        samples_id2name[i] = samples[i]
        samples_name2id[samples[i]] = i

    sample_dir = Path(cfg.dataset.base_dir) / cfg.dataset.name / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_dir / "samples.pickle", "wb") as f:
        pickle.dump(samples_id2name, f)
    with open(sample_dir / "samples_reverse.pickle", "wb") as f:
        pickle.dump(samples_name2id, f)
    return
