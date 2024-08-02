import os
import glob
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

from base import Config


# TODO: remove config arg


def create_actions(cfg: Config):
    gt_files = glob.glob(
        f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.gt_dir}/*.txt"
    )
    os.makedirs(f"{cfg.dataset.base_dir}/{cfg.dataset.name}/actions", exist_ok=True)

    for gt in tqdm(gt_files, leave=False):
        if Path(f"{cfg.dataset.base_dir}/{cfg.dataset.name}/actions/{Path(gt).name}").exists():
            continue

        with open(gt, "r") as f:
            lines = f.readlines()
            actions = OrderedDict.fromkeys(
                [
                    line.strip()
                    for line in lines
                    if line.strip() not in cfg.dataset.backgrounds
                ]
            )

        with open(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/actions/{Path(gt).name}", "w"
        ) as f:
            f.writelines([f"{action}\n" for action in actions])
