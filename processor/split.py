import os
import glob
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

from utils import init_seed
from loader import get_actions
from manager import Config


def shuffle_split(cfg: Config):
    init_seed(cfg.seed)
    actions = get_actions(f"{cfg.base_dir}/{cfg.dataset}/actions.txt")
    kfold = KFold(n_splits=cfg.num_fold, shuffle=True, random_state=cfg.seed)
    os.makedirs(f"{cfg.base_dir}/{cfg.dataset}/{cfg.split_dir}", exist_ok=True)

    for action in actions.keys():
        files = glob.glob(f"{cfg.base_dir}/{cfg.dataset}/{cfg.gt_dir}/*{action}*.txt")
        files.sort()
        files = np.random.choice(
            files, int(len(files) * cfg.semi_per), replace=False
        ).tolist()
        train_paths = []
        test_paths = []

        for i, (train, test) in enumerate(kfold.split(files)):
            train_paths = [Path(files[i]).name + "\n" for i in train]
            train_paths.sort()
            test_paths = [Path(files[i]).name + "\n" for i in test]
            test_paths.sort()

            with open(
                f"{cfg.base_dir}/{cfg.dataset}/{cfg.split_dir}/train.split{i+1}.{cfg.semi_per:.2f}.bundle",
                "a",
            ) as f:
                f.writelines(train_paths)

            with open(
                f"{cfg.base_dir}/{cfg.dataset}/{cfg.split_dir}/test.split{i+1}.{cfg.semi_per:.2f}.bundle",
                "a",
            ) as f:
                f.writelines(test_paths)
