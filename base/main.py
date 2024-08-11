import time
import glob
from pathlib import Path
import random
from dataclasses import asdict, astuple, dataclass
import cv2
import numpy as np
import torch
from torch.nn import Module
from omegaconf import DictConfig

from logger import Logger

from numpy import ndarray
from torch import Tensor


@dataclass
class DatasetConfig(DictConfig):
    name: str
    mapping_path: str | None
    actions_path: str | None
    matching_path: str | None
    has_mapping_header: bool | None
    mapping_separator: str | None
    has_actions_header: bool | None
    actions_action_separator: str | None
    actions_class_separator: str | None
    matching_separator: str | None
    num_classes: int
    input_dim: int
    split: int
    num_fold: int
    backgrounds: list[str]
    batch_size: int
    sampling_rate: int
    shuffle: bool
    base_dir: str
    split_dir: str
    gt_dir: str
    feature_dir: str
    prob_dir: str | None
    pseudo_dir: str | None
    semi_per: float | None


@dataclass
class Config(DictConfig):
    # system
    seed: int
    device: str
    verbose: bool
    val_skip: bool

    dataset: DatasetConfig

    # learning
    train: bool
    model_name: str
    best_model_path: str
    result_dir: str
    epochs: int
    lr: float
    weight_decay: float
    use_pseudo: bool
    refine_pseudo: bool

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __post_init__(self):
        self.dataset = DatasetConfig(**asdict(self.dataset))

    def __iter__(self):
        return iter(astuple(self))

    def __or__(self, other):
        return self.__class__(**asdict(self) | asdict(other))


class Base:
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

    @staticmethod
    def init_seed(seed: int = 42):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

    @staticmethod
    def get_device(cuda: int | str = 0) -> torch.device:
        return torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def info(message: str):
        Logger.log(message)

    @staticmethod
    def warning(message: str):
        Logger.log(message, level="warning")

    @staticmethod
    def error(message: str):
        Logger.log(message, level="error")

    @staticmethod
    def get_logger(name: str = "Base") -> Logger:
        return Logger(name)

    @staticmethod
    def get_time() -> float:
        return time.time()

    @staticmethod
    def get_elapsed_time(start: float) -> float:
        return time.time() - start

    @staticmethod
    def log_time(logger, seconds: float):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        logger.info(f"Elapsed time: {hours:02d}h {minutes:02d}m {seconds:.0f}s")

    @staticmethod
    def validate_config(cfg: Config):
        return True

    @staticmethod
    def load_model(
        model: Module, model_path: str, device: torch.device | str = "cpu", logger=None
    ):
        model = model.to(device)
        _model_path = Path(model_path)
        if _model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            if logger:
                logger.warning(f"Model was not found in {model_path}")
            else:
                print(f"Model was not found in {model_path}")
        return model

    @staticmethod
    def load_best_model(
        model: Module, model_dir: str, device: torch.device | str = "cpu", logger=None
    ):
        model_paths = glob.glob(f"{model_dir}/*")
        model_paths.sort()
        best_model = next(filter(lambda x: "best" in x, model_paths), None)
        if best_model:
            model = Base.load_model(model, best_model, device)
        else:
            if logger:
                logger.warning("Best model was not found")
            else:
                print("Best model was not found")

        return model

    @staticmethod
    def save_model(model: Module, model_path: str):
        model = model.cpu()
        torch.save(model.state_dict(), model_path)

    @staticmethod
    def to_np(x: list | ndarray | Tensor | None) -> ndarray:
        if x is None or len(x) == 0:
            return np.array([])
        if isinstance(x, list):
            if isinstance(x[0], str):
                return Base.to_class_index(x)
        if isinstance(x, ndarray):
            return x
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy()
        raise ValueError("Invalid input type")

    @staticmethod
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

    @staticmethod
    def to_class_name(x: ndarray | Tensor) -> list[str]:
        _x = Base.to_np(x)
        return [Base.int_to_text[i] for i in _x]

    @staticmethod
    def to_class_index(x: list[str]) -> ndarray:
        _x = [Base.text_to_int[i] for i in x]
        return Base.to_np(_x)

    @staticmethod
    def get_image_paths(img_dir: str) -> list[str]:
        image_paths = glob.glob(f"{img_dir}/*.png")
        image_paths += glob.glob(f"{img_dir}/*.jpg")
        image_paths.sort()
        return image_paths

    @staticmethod
    def load_image(img_path: str) -> ndarray:
        return cv2.imread(img_path)

    @staticmethod
    def load_images(img_dir: str) -> list[ndarray]:
        image_paths = Base.get_image_paths(img_dir)
        return [Base.load_image(img_path) for img_path in image_paths]

    @staticmethod
    def get_class_mapping(
        mapping_path: str, has_header: bool = False, separator: str = " "
    ) -> tuple[dict[str, int], dict[int, str]]:
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
        return text_to_int, int_to_text

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

    @staticmethod
    def get_actions(
        actions_path: str,
        has_header: bool = False,
        action_separator: str = " ",
        class_separator: str = ",",
        text_to_int: dict[str, int] = dict(),
    ) -> dict[str, list[int]]:
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
                text_to_int[c] for c in classes.split(class_separator) if c != ""
            ]
        return actions

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

    @staticmethod
    def get_action_matching(matching_path: str, separator: str = " ") -> dict[str, str]:
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
        return video_to_action

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

    @staticmethod
    def to_segments(x: ndarray):
        diff = np.diff(x, prepend=-100)
        indices = np.where(diff != 0)[0]
        segments = []

        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]
            segments.append((x[start], (start, end)))
        segments.append((x[indices[-1]], (indices[-1], len(x))))

        return segments

    @staticmethod
    def mask_label_with_backgrounds(x: ndarray, backgrounds: ndarray):
        if len(backgrounds) == 0:
            return x
        return np.array([t if t not in backgrounds else backgrounds[0] for t in x])
