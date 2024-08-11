import os
import glob
import math
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

from base import Base, Config

# TODO: remove config arg


def shuffle_split(cfg: Config):
    Base.init_seed(cfg.seed)
    text_to_int, _ = Base.get_class_mapping(
        f"{cfg.dataset.base_dir}/{cfg.dataset.name}/mapping.txt"
    )
    actions = Base.get_actions(
        f"{cfg.dataset.base_dir}/{cfg.dataset.name}/actions.txt",
        text_to_int=text_to_int,
    )
    kfold = KFold(n_splits=cfg.dataset.num_fold, shuffle=True, random_state=cfg.seed)
    os.makedirs(
        f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.split_dir}",
        exist_ok=True,
    )
    matching = Base.get_action_matching(
        f"{cfg.dataset.base_dir}/{cfg.dataset.name}/matching.txt"
    )

    for action in actions.keys():
        if cfg.dataset.name == "nissan":
            if action == "no_action":
                continue
            files = glob.glob(
                f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.gt_dir}/*.txt"
            )
            files = [
                f
                for f in files
                if Path(f).name in matching.keys() and matching[Path(f).name] == action
            ]
        elif cfg.dataset.name == "salads":
            files = glob.glob(
                f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.gt_dir}/*.txt"
            )
        else:
            files = glob.glob(
                f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.gt_dir}/*{action}*.txt"
            )

        files.sort()
        if len(files) < cfg.dataset.num_fold:
            Base.warning(f"Action {action} has less than {cfg.dataset.num_fold} files")
            continue

        for i, (train, test) in enumerate(kfold.split(files)):
            train_paths = [Path(files[j]).name + "\n" for j in train]
            train_paths.sort()
            test_paths = [Path(files[j]).name + "\n" for j in test]
            test_paths.sort()

            semi_per = cfg.dataset.semi_per or 1.0
            file_num = math.ceil(len(train_paths) * semi_per)
            train_paths = np.random.choice(
                train_paths, file_num, replace=False
            ).tolist()

            with open(
                f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.split_dir}/train.split{i+1}.{cfg.dataset.semi_per:.2f}.bundle",
                "a",
            ) as f:
                f.writelines(train_paths)

            with open(
                f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.split_dir}/test.split{i+1}.{cfg.dataset.semi_per:.2f}.bundle",
                "a",
            ) as f:
                f.writelines(test_paths)
