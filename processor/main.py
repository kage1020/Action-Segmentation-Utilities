from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from ..base import (
    Base,
    Config,
    get_action_matching,
    get_actions,
    get_class_mapping,
    get_gt,
    init_seed,
    load_file,
    save_file,
)
from ..logger import log


class Processor(Base):
    def __init__(self):
        super().__init__(name="Processor")

    @staticmethod
    def shuffle_split(cfg: Config):
        init_seed(cfg.seed)
        text_to_int, _ = get_class_mapping(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/mapping.txt"
        )
        actions = get_actions(
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
        matching = get_action_matching(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/matching.txt"
        )
        gt_dir = Path(f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.gt_dir}")

        for action in tqdm(actions.keys(), leave=False, desc="Actions"):
            if cfg.dataset.name == "nissan":
                if action == "no_action":
                    continue
                files = list(gt_dir.glob("*.txt"))
                files = [
                    f
                    for f in files
                    if f.name in matching.keys() and matching[f.name] == action
                ]
            elif cfg.dataset.name == "salads":
                files = list(gt_dir.glob("*.txt"))
            else:
                files = list(gt_dir.glob(f"*{action}*.txt"))

            files.sort()
            if len(files) < cfg.dataset.num_fold:
                log(
                    f"Action {action} has less than {cfg.dataset.num_fold} files",
                    level="warning",
                )
                continue

            for i, (train, test) in enumerate(
                tqdm(kfold.split(files), leave=False, desc="K-Folds")
            ):
                train_files = [files[j].name + "\n" for j in train]
                train_files.sort()
                test_files = [files[j].name + "\n" for j in test]
                test_files.sort()

                semi_per = cfg.dataset.semi_per or 1.0
                file_num = np.ceil(len(train_files) * semi_per)
                train_paths = np.random.choice(
                    train_files, file_num, replace=False
                ).tolist()

                with open(
                    split_dir / f"train.split{i+1}.{semi_per:.2f}.bundle",
                    "a",
                ) as f:
                    f.writelines(train_paths)

                with open(
                    split_dir / f"test.split{i+1}.{semi_per:.2f}.bundle",
                    "a",
                ) as f:
                    f.writelines(test_files)

    @staticmethod
    def analytics_split(cfg: Config):
        text_to_int, int_to_text = get_class_mapping(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/mapping.txt"
        )
        split_dir = Path(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.split_dir}"
        )
        matching = get_action_matching(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/matching.txt"
        )
        gt_dir = Path(f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.gt_dir}")

        split_stats = {}
        action_stats = {}
        class_stats = {}

        split_files = list(split_dir.glob("*.bundle"))
        semi_per = cfg.dataset.semi_per or 1.0
        split_files = [f for f in split_files if f"{semi_per:.2f}" in f.name]
        for split_file in split_files:
            gt_paths = load_file(split_file)
            train_files = [f for f in gt_paths if f in matching.keys()]
            test_files = [f for f in gt_paths if f not in matching.keys()]

            train_gts = [
                get_gt(str(gt_dir / f), text_to_int=text_to_int) for f in train_files
            ]
            test_gts = [
                get_gt(str(gt_dir / f), text_to_int=text_to_int) for f in test_files
            ]
            train_frames = sum(len(g) for g in train_gts)
            test_frames = sum(len(g) for g in test_gts)

            split_stats[split_file.name] = {
                "train": {"count": len(train_files), "time": train_frames},
                "test": {"count": len(test_files), "time": test_frames},
            }
            for p, a in zip(train_files, train_gts):
                action_stats[matching[p]] = {
                    "train": {
                        "count": 1
                        + action_stats.get(matching[p], {})
                        .get("train", {})
                        .get("count", 0),
                        "time": len(a)
                        + action_stats.get(matching[p], {})
                        .get("train", {})
                        .get("time", 0),
                    },
                    "test": {
                        "count": 1
                        + action_stats.get(matching[p], {})
                        .get("test", {})
                        .get("count", 0),
                        "time": len(a)
                        + action_stats.get(matching[p], {})
                        .get("test", {})
                        .get("time", 0),
                    },
                }
            for gt in train_gts:
                for c in gt:
                    class_stats[int_to_text[c]] = {
                        "train": {
                            "time": 1
                            + class_stats.get(int_to_text[c], {})
                            .get("train", {})
                            .get("time", 0),
                        },
                        "test": {
                            "time": 1
                            + class_stats.get(int_to_text[c], {})
                            .get("test", {})
                            .get("time", 0),
                        },
                    }
        split_stats["total"] = {
            "train": {
                "count": sum(s["train"]["count"] for s in split_stats.values()),
                "time": sum(s["train"]["time"] for s in split_stats.values()),
            },
            "test": {
                "count": sum(s["test"]["count"] for s in split_stats.values()),
                "time": sum(s["test"]["time"] for s in split_stats.values()),
            },
        }
        action_stats["total"] = {
            "train": {
                "count": sum(a["train"]["count"] for a in action_stats.values()),
                "time": sum(a["train"]["time"] for a in action_stats.values()),
            },
            "test": {
                "count": sum(a["test"]["count"] for a in action_stats.values()),
                "time": sum(a["test"]["time"] for a in action_stats.values()),
            },
        }
        class_stats["total"] = {
            "train": {
                "time": sum(c["train"]["time"] for c in class_stats.values()),
            },
            "test": {
                "time": sum(c["test"]["time"] for c in class_stats.values()),
            },
        }

        for split in split_stats.keys():
            split_stats[split]["train"]["time"] /= 25 * 60
            split_stats[split]["test"]["time"] /= 25 * 60
        for action in action_stats.keys():
            action_stats[action]["train"]["time"] /= 25 * 60
            action_stats[action]["test"]["time"] /= 25 * 60
        for c in class_stats.keys():
            class_stats[c]["train"]["time"] /= 25 * 60
            class_stats[c]["test"]["time"] /= 25 * 60

        save_file(
            "data/nissan/analytics.md",
            [
                "## Split stats",
                "",
                "| split   | video | time (min.) |",
                "|---------|-------|-------------|",
                *[
                    f"| {split} | {split_stats[split]['train']['count']} | {split_stats[split]['train']['time']:.2f} |"
                    for split in split_stats.keys()
                ],
                "",
                "## Action stats",
                "",
                "| action | video | time (min.) |",
                "|--------|-------|-------------|",
                *[
                    f"| {action} | {action_stats[action]['train']['count']} | {action_stats[action]['train']['time']:.2f} |"
                    for action in action_stats.keys()
                ],
                "",
                "## Class stats",
                "",
                "| class | time (min.) |",
                "|-------|-------------|",
                *[
                    f"| {c} | {class_stats[c]['train']['time']:.2f} |"
                    for c in class_stats.keys()
                ],
            ],
        )

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
