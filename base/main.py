import glob
import random
from dataclasses import dataclass
import cv2
import numpy as np
import torch
from torch.nn import Module

from numpy import ndarray
from torch import Tensor


@dataclass()
class Config:
    device: str


class Base:
    backgrounds: list[int]
    text_to_int: dict[str, int] = dict()
    int_to_text: dict[int, str] = dict()
    actions: dict[str, list[int]] = dict()
    video_to_action: dict[str, str] = dict()

    def __init__(
        self,
        mapping_path: str | None = None,
        actions_path: str | None = None,
        matching_path: str | None = None,
        has_mapping_header: bool = False,
        mapping_separator: str = " ",
        has_actions_header: bool = False,
        actions_action_separator: str = " ",
        actions_class_separator: str = ",",
        matching_separator: str = " ",
        backgrounds: list[str] = [],
    ):
        if mapping_path is not None:
            self.set_class_mapping(
                mapping_path, has_header=has_mapping_header, separator=mapping_separator
            )
            self.backgrounds = [self.text_to_int[c] for c in backgrounds]
        if actions_path is not None:
            self.set_actions(
                actions_path,
                has_header=has_actions_header,
                action_separator=actions_action_separator,
                class_separator=actions_class_separator,
            )
        if matching_path is not None:
            self.set_action_matching(matching_path, separator=matching_separator)

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
    def load_model(model: Module, model_path: str, device="cpu"):
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    @staticmethod
    def load_best_model(model: Module, model_dir: str, device="cpu"):
        model_paths = glob.glob(f"{model_dir}/*")
        model_paths.sort()
        best_model = next(filter(lambda x: "best" in x, model_paths), None)
        if best_model:
            model = Base.load_model(model, best_model, device)
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
            action, classes = line.split(action_separator)
            actions[action] = [
                self.text_to_int[c] for c in classes.split(class_separator)
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
        assert len(backgrounds) > 0, "backgrounds should have at least one element"
        return np.array([t if t not in backgrounds else backgrounds[0] for t in x])
