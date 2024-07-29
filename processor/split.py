import os
import glob
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

from base import Base, Config


def shuffle_split(cfg: Config):
    Base.init_seed(cfg.seed)
    actions = Base.get_actions(f"{cfg.dataset.base_dir}/{cfg.dataset}/actions.txt")
    kfold = KFold(n_splits=cfg.dataset.num_fold, shuffle=True, random_state=cfg.seed)
    os.makedirs(
        f"{cfg.dataset.base_dir}/{cfg.dataset}/{cfg.dataset.split_dir}", exist_ok=True
    )

    for action in actions.keys():
        files = glob.glob(
            f"{cfg.dataset.base_dir}/{cfg.dataset}/{cfg.dataset.gt_dir}/*{action}*.txt"
        )
        files.sort()
        semi_per = cfg.dataset.semi_per or 1.0
        files = np.random.choice(
            files, int(len(files) * semi_per), replace=False
        ).tolist()
        train_paths = []
        test_paths = []

        for i, (train, test) in enumerate(kfold.split(files)):
            train_paths = [Path(files[i]).name + "\n" for i in train]
            train_paths.sort()
            test_paths = [Path(files[i]).name + "\n" for i in test]
            test_paths.sort()

            with open(
                f"{cfg.dataset.base_dir}/{cfg.dataset}/{cfg.dataset.split_dir}/train.split{i+1}.{cfg.dataset.semi_per:.2f}.bundle",
                "a",
            ) as f:
                f.writelines(train_paths)

            with open(
                f"{cfg.dataset.base_dir}/{cfg.dataset}/{cfg.dataset.split_dir}/test.split{i+1}.{cfg.dataset.semi_per:.2f}.bundle",
                "a",
            ) as f:
                f.writelines(test_paths)
