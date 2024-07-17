import os
import glob
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

# add config type


def create_actions(cfg):
    gt_files = glob.glob(f"{cfg.base_dir}/{cfg.dataset}/{cfg.gt_dir}/*.txt")
    os.makedirs(f"{cfg.base_dir}/{cfg.dataset}/actions", exist_ok=True)

    for gt in tqdm(gt_files, leave=False):
        with open(gt, "r") as f:
            lines = f.readlines()
            actions = OrderedDict.fromkeys(
                [line.strip() for line in lines if line.strip() not in cfg.backgrounds]
            )

        with open(f"{cfg.base_dir}/{cfg.dataset}/actions/{Path(gt).name}", "w") as f:
            f.writelines([f"{action}\n" for action in actions])
