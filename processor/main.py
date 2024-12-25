from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np

from base.main import Base, Config


class Processor(Base):
    def __init__(self):
        super().__init__(name="Processor")

    @staticmethod
    def shuffle_split(cfg: Config):
        Base.init_seed(cfg.seed)
        text_to_int, _ = Base.get_class_mapping(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/mapping.txt"
        )
        actions = Base.get_actions(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/actions.txt",
            text_to_int=text_to_int,
        )
        kfold = KFold(
            n_splits=cfg.dataset.num_fold, shuffle=True, random_state=cfg.seed
        )
        split_dir = Path(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.split_dir}"
        )
        split_dir.mkdir(parents=True, exist_ok=True)
        matching = Base.get_action_matching(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/matching.txt"
        )
        gt_dir = Path(f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.gt_dir}")

        for action in actions.keys():
            if cfg.dataset.name == "nissan":
                if action == "no_action":
                    continue
                files = list(gt_dir.glob("*.txt"))
                files = [
                    f
                    for f in files
                    if f.stem in matching.keys() and matching[f.stem] == action
                ]
            elif cfg.dataset.name == "salads":
                files = list(gt_dir.glob("*.txt"))
            else:
                files = list(gt_dir.glob(f"*{action}*.txt"))

            files.sort()
            if len(files) < cfg.dataset.num_fold:
                Base.warning(
                    f"Action {action} has less than {cfg.dataset.num_fold} files"
                )
                continue

            for i, (train, test) in enumerate(kfold.split(files)):
                train_paths = [files[j].name + "\n" for j in train]
                train_paths.sort()
                test_paths = [files[j].name + "\n" for j in test]
                test_paths.sort()

                semi_per = cfg.dataset.semi_per or 1.0
                file_num = np.ceil(len(train_paths) * semi_per)
                train_paths = np.random.choice(
                    train_paths, file_num, replace=False
                ).tolist()

                with open(
                    split_dir / f"train.split{i+1}.{cfg.dataset.semi_per:.2f}.bundle",
                    "a",
                ) as f:
                    f.writelines(train_paths)

                with open(
                    split_dir / f"test.split{i+1}.{cfg.dataset.semi_per:.2f}.bundle",
                    "a",
                ) as f:
                    f.writelines(test_paths)

    @staticmethod
    def create_action_gt(cfg: Config):
        gt_dir = Path(f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.gt_dir}")
        gt_files = list(gt_dir.glob("*.txt"))
        out_dir = Path(f"{cfg.dataset.base_dir}/{cfg.dataset.name}/actions")
        out_dir.mkdir(parents=True, exist_ok=True)

        for gt in tqdm(gt_files, leave=False):
            if Path(
                f"{cfg.dataset.base_dir}/{cfg.dataset.name}/actions/{Path(gt).name}"
            ).exists():
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
                f"{cfg.dataset.base_dir}/{cfg.dataset.name}/actions/{Path(gt).name}",
                "w",
            ) as f:
                f.writelines([f"{action}\n" for action in actions])
