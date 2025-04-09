import glob
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from numpy import ndarray
from torch import Tensor
from torch.nn import Module

from configs import Config
from logger import Logger, log

load_dotenv()


def unique(
    x: list | ndarray, return_index: bool = False
) -> ndarray | tuple[ndarray, ndarray]:
    _, unique_indices = np.unique(x, return_index=True)
    if return_index:
        return np.array([x[i] for i in np.sort(unique_indices)]), unique_indices
    return np.array([x[i] for i in np.sort(unique_indices)])


def init_seed(seed: int = 42):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def get_device(cuda: int | str = 0) -> torch.device:
    return torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")


def get_time() -> float:
    return time.time()


def get_elapsed_time(start: float) -> float:
    return time.time() - start


def log_time(seconds: float, logger: Logger = Logger()):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    logger.info(f"Elapsed time: {hours:02d}h {minutes:02d}m {seconds:.0f}s")


def get_dirs(path: str | Path, recursive: bool = False) -> list[Path]:
    path = str(path)
    dirs = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]
    if not recursive:
        return [Path(d) for d in dirs]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(lambda d: get_dirs(d, recursive), dirs))
        for result in results:
            dirs.extend(result)
    return [Path(d) for d in dirs]


def create_config(cfg: dict) -> Config:
    return Config(**cfg)


def validate_config(cfg: Config):
    log("Config validation is not implemented", "warning")
    return True


def load_model(
    model: Module,
    model_path: str,
    device: torch.device | str = "cpu",
    logger=None,
    strict: bool = True,
):
    if not logger:
        logger = Logger()
    model = model.to(device)
    _model_path = Path(model_path)
    if _model_path.exists() and _model_path.is_file():
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True),
            strict=strict,
        )
        logger.info(f"Model was loaded from {model_path}")
    else:
        logger.warning(f"Model was not found in {model_path}")
    return model


def load_best_model(
    model: Module, model_dir: str, device: torch.device | str = "cpu", logger=None
):
    model_paths = glob.glob(f"{model_dir}/*")
    model_paths.sort()
    best_model = next(filter(lambda x: "best" in x, model_paths), None)
    if best_model:
        model = load_model(model, best_model, device)
    else:
        if logger:
            logger.warning("Best model was not found")
        else:
            print("Best model was not found")

    return model


def save_model(model: Module, model_path: str):
    model = model.cpu()
    torch.save(model.state_dict(), model_path)


def to_np(x: list | ndarray | Tensor, mapping: dict[str, int] | None = None) -> ndarray:
    if len(x) == 0:
        return np.array([])
    if isinstance(x, list):
        if isinstance(x[0], str):
            return to_class_index(x, mapping if mapping is not None else {})
        return np.array(x)
    if isinstance(x, ndarray):
        return x
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    raise ValueError("Invalid input type")


def to_torch(x: list | ndarray | Tensor | None) -> Tensor:
    if x is None:
        return Tensor([])
    if isinstance(x, list):
        return Tensor(x)
    if isinstance(x, ndarray):
        return Tensor(x)
    if isinstance(x, Tensor):
        return x.detach()
    raise ValueError("Invalid input type")


def save_np(x: ndarray, path: str) -> None:
    np.save(path, x)


def save_labels(x: list, path: str) -> None:
    with open(path, "w") as f:
        f.writelines([f"{item}\n" for item in x])


def to_class_name(x: ndarray | Tensor, int_to_text: dict[int, str]) -> list[str]:
    _x = to_np(x)
    return [int_to_text[i] for i in _x]


def to_class_index(x: list[str], text_to_int: dict[str, int]) -> ndarray:
    _x = [text_to_int[i] for i in x]
    return to_np(_x)


def get_image_paths(img_dir: str) -> list[str]:
    image_paths = glob.glob(f"{img_dir}/*.png")
    image_paths += glob.glob(f"{img_dir}/*.jpg")
    image_paths.sort()
    return image_paths


def load_file(path: str | Path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def save_file(path: str | Path, x: list[str]):
    with open(path, "w") as f:
        f.write("\n".join(x))


def load_image(img_path: str) -> ndarray:
    return cv2.imread(img_path)


def load_images(img_dir: str) -> list[ndarray]:
    image_paths = get_image_paths(img_dir)
    return [load_image(img_path) for img_path in image_paths]


def to_image(feature: ndarray) -> ndarray:
    _feature = feature - feature.min()
    _feature = _feature / _feature.max()
    return (_feature * 255).astype(np.uint8)


def get_class_mapping(
    mapping_path: str, has_header: bool = False, separator: str = " "
) -> tuple[dict[str, int], dict[int, str]]:
    try:
        with open(mapping_path, "r") as f:
            lines = f.readlines()
        if has_header:
            lines = lines[1:]
    except FileNotFoundError:
        log(f"Mapping file was not found in {mapping_path}", level="warning")
        return {}, {}

    text_to_int = {}
    int_to_text = {}
    for line in lines:
        text, idx = line.strip().split(separator)
        try:
            idx = int(idx)
        except ValueError:
            text, idx = idx, int(text)
        text_to_int[text] = idx
        int_to_text[idx] = text
    return text_to_int, int_to_text


def get_gt(gt_path: str, text_to_int: dict[str, int] = dict()) -> list[int]:
    try:
        with open(gt_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip() != ""]
            lines = [text_to_int[line] for line in lines]
    except FileNotFoundError:
        log(f"GT file was not found in {gt_path}", level="warning")
        return []

    return [int(line) for line in lines]


def get_actions(
    actions_path: str,
    has_header: bool = False,
    action_separator: str = " ",
    class_separator: str = ",",
    text_to_int: dict[str, int] = dict(),
) -> dict[str, list[int]]:
    try:
        with open(actions_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip() != ""]
        if has_header:
            lines = lines[1:]
    except FileNotFoundError:
        log(f"Actions file was not found in {actions_path}", level="warning")
        return {}

    actions = {}
    for line in lines:
        try:
            action, classes = line.split(action_separator)
        except ValueError:
            action = line
            classes = ""
        actions[action] = [
            text_to_int[c] for c in classes.split(class_separator) if c != ""
        ]
    return actions


def get_action_matching(matching_path: str, separator: str = " ") -> dict[str, str]:
    """
    Get the matching of classes

    Matching file format should be:
    video1 action1
    video2 action2
    ...
    """
    try:
        with open(matching_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip() != ""]
    except FileNotFoundError:
        log(f"Matching file was not found in {matching_path}", level="warning")
        return {}
    video_to_action = {}
    for line in lines:
        video, action = line.split(separator)
        video_to_action[video] = action
    return video_to_action


def get_boundaries(boundary_dir: Path) -> dict[str, list[tuple[int, int]]]:
    """
    Get the boundaries of the videos

    Boundary file format should be:
    0    100
    101  200
    ...
    """
    boundary_files = list(boundary_dir.glob("*.txt"))
    boundary_files.sort()
    boundaries = {}

    for boundary_file in boundary_files:
        with open(boundary_file, "r") as f:
            lines = f.readlines()
        boundaries[boundary_file.stem] = [
            [int(x) for x in line.strip().split()] for line in lines
        ]
    return boundaries


def to_segments(
    x: ndarray, backgrounds: ndarray = np.array([])
) -> list[tuple[int, tuple[int, int]]]:
    _x = np.array(x)
    diff = np.diff(_x, prepend=-100)
    indices = np.where(diff != 0)[0]
    segments = []

    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i + 1]
        if _x[start] not in backgrounds:
            segments.append((_x[start], (start, end)))
    if _x[indices[-1]] not in backgrounds:
        segments.append((_x[indices[-1]], (indices[-1], len(_x))))

    return segments


def mask_label_with_backgrounds(x: ndarray, backgrounds: ndarray):
    if len(backgrounds) == 0:
        return x
    return np.array([t if t not in backgrounds else backgrounds[0] for t in x])


def get_anomalies(anomaly_dir: Path) -> dict[str, list[str]]:
    """
    Get the anomalies of the videos

    Anomaly file format should be:
    normality
    normality
    anomaly
    ...
    """
    anomaly_files = list(anomaly_dir.glob("*.txt"))
    anomaly_files.sort()
    anomalies = {}

    for anomaly_file in anomaly_files:
        with open(anomaly_file, "r") as f:
            lines = f.readlines()
        anomalies[anomaly_file.stem] = [line.strip() for line in lines]
    return anomalies


class Env:
    def __getattr__(self, key):
        return os.getenv(key) == "True"


class Base:
    env = Env()
    backgrounds: list[int]
    text_to_int: dict[str, int] = dict()
    int_to_text: dict[int, str] = dict()
    actions: dict[str, list[int]] = dict()
    video_to_action: dict[str, str] = dict()

    def __init__(
        self,
        name: str = "Base",
        mapping_path: str | None = None,
        actions_path: str | None = None,
        matching_path: str | None = None,
        has_mapping_header: bool | None = False,
        mapping_separator: str | None = " ",
        has_actions_header: bool | None = False,
        actions_action_separator: str | None = " ",
        actions_class_separator: str | None = ",",
        matching_separator: str | None = " ",
        backgrounds: list[str] = [],
    ):
        self.logger = Logger(name)

        if mapping_path is not None:
            self.set_class_mapping(
                mapping_path,
                has_header=has_mapping_header or False,
                separator=mapping_separator or " ",
            )
            self.backgrounds = [self.text_to_int[c] for c in backgrounds]
        if actions_path is not None:
            self.set_actions(
                actions_path,
                has_header=has_actions_header or False,
                action_separator=actions_action_separator or " ",
                class_separator=actions_class_separator or ",",
            )
        if matching_path is not None:
            self.set_action_matching(matching_path, separator=matching_separator or " ")

    def set_class_mapping(
        self, mapping_path: str, has_header: bool = False, separator: str = " "
    ) -> None:
        """
        Get the mapping of classes to integers

        Mapping file format should be:
        class1 0
        class2 1
        """
        with open(mapping_path, "r") as f:
            lines = f.readlines()
        if has_header:
            lines = lines[1:]

        text_to_int = {}
        int_to_text = {}
        for line in lines:
            text, idx = line.strip().split(separator)
            try:
                idx = int(idx)
            except ValueError:
                text, idx = idx, int(text)
            text_to_int[text] = idx
            int_to_text[idx] = text
        self.text_to_int = text_to_int
        self.int_to_text = int_to_text

    def update_class_mapping(self, x: list[str]) -> None:
        for i in x:
            if i not in self.text_to_int:
                idx = len(self.text_to_int)
                self.text_to_int[i] = idx
                self.int_to_text[idx] = i

    def mask_mapping_with_backgrounds(self, mask_value: int = -100) -> None:
        for c in self.backgrounds:
            self.text_to_int[self.int_to_text[c]] = mask_value

    def set_actions(
        self,
        actions_path: str,
        has_header: bool = False,
        action_separator: str = " ",
        class_separator: str = ",",
    ) -> None:
        """
        Get the actions per video

        Actions file format should be:
        action1 class1,class2,class3,...
        action2 class4,class5,class6,...
        """
        assert (
            len(self.text_to_int) > 1
        ), "Class mapping is not set. Please set the class mapping first by `set_class_mapping` method."

        with open(actions_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip() != ""]
        if has_header:
            lines = lines[1:]

        actions = {}
        for line in lines:
            try:
                action, classes = line.split(action_separator)
            except ValueError:
                action = line
                classes = ""
            actions[action] = [
                self.text_to_int[c] for c in classes.split(class_separator) if c != ""
            ]
        self.actions = actions

    def mask_actions_with_backgrounds(self, mask_value=-100) -> None:
        for action in self.actions.keys():
            self.actions[action] = [
                mask_value if c in self.backgrounds else c for c in self.actions[action]
            ]

    def set_action_matching(self, matching_path: str, separator: str = " ") -> None:
        """
        Get the matching of classes

        Matching file format should be:
        video1 action1
        video2 action2
        ...
        """
        with open(matching_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip() != ""]
        video_to_action = {}
        for line in lines:
            video, action = line.split(separator)
            video_to_action[video] = action
        self.video_to_action = video_to_action
