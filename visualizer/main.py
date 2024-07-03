import os
import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import cv2
import numpy as np
from torch import Tensor
from .writer import VideoWriter
from .palette import template


def get_mapping(file_path: str, has_header: bool = False):
    """
    Get the mapping of classes to integers

    Mapping file format should be:
    class1 0
    class2 1
    ...
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
        if has_header:
            lines = lines[1:]
    text_to_int = {}
    int_to_text = {}
    is_csv = "," in lines[0]
    for line in lines:
        if is_csv:
            text, num = line.strip().split(",")
        else:
            text, num = line.strip().split()
        try:
            num = int(num)
        except ValueError:
            text, num = num, int(text)
        text_to_int[text] = num
        int_to_text[num] = text
    return text_to_int, int_to_text


class Visualizer:
    def __init__(
        self, backgrounds: list[str] | np.ndarray | Tensor = [], num_classes: int = 50
    ):
        self.backgrounds = backgrounds
        self.num_classes = num_classes
        self.pred = None
        self.gt = None
        self.confidences = None

    @staticmethod
    def load_images(image_dir: str) -> list[str]:
        image_paths = glob.glob(f"{image_dir}/*.png")
        image_paths += glob.glob(f"{image_dir}/*.jpg")
        image_paths.sort()
        return image_paths

    @staticmethod
    def load_image(image: str | np.ndarray):
        if isinstance(image, str):
            return cv2.imread(image)
        if isinstance(image, np.ndarray):
            return image

    @staticmethod
    def to_np(
        x: list | np.ndarray | Tensor, mapping: dict[str, int] | None = None
    ) -> np.ndarray:
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy(), mapping
        if isinstance(x, np.ndarray):
            return x, mapping
        if isinstance(x, list):
            if len(x) > 0 and isinstance(x[0], str):
                x, _mapping = Visualizer.str_to_int(x, mapping)
                return np.array(x), _mapping
            return np.array(x), mapping
        raise ValueError("Invalid input type")

    @staticmethod
    def to_segments(
        x: list[str] | np.ndarray | Tensor,
        backgrounds: list[str] | np.ndarray | Tensor,
        mapping: dict[str, int] | None = None,
    ) -> list[int, tuple[int, int]]:
        _x, _mapping = Visualizer.to_np(x, mapping)
        _backgrounds, _mapping = Visualizer.to_np(backgrounds, mapping)
        diff = np.diff(_x, prepend=-1)
        indices = np.where(diff != 0)[0]
        segments = []

        for i in range(len(indices)):
            if _x[indices[i]] in _backgrounds:
                continue
            start = indices[i]
            end = indices[i + 1] if i + 1 < len(indices) else len(_x)
            segments.append((_x[start], (start, end)))

        return segments, _mapping

    @staticmethod
    def str_to_int(x: list[str], mapping: dict[str, int] | None = dict()):
        _mapping = mapping.copy()
        for s in x:
            if s not in _mapping:
                _mapping[s] = len(_mapping)
        return [_mapping[s] for s in x], _mapping

    @staticmethod
    def plot_feature(feature: np.ndarray, file_path: str = "feature.png"):
        assert isinstance(feature, np.ndarray) or isinstance(
            feature, Tensor
        ), "Feature must be a numpy array or a torch tensor"
        assert len(feature.shape) == 2, "Feature must be a 2D array"

        fig = plt.figure()
        ax = fig.add_subplot(111)
        axfig = ax.imshow(feature, aspect="auto", interpolation="none", cmap="jet")
        fig.colorbar(axfig, ax=ax)
        fig.subplots_adjust(left=0.1, right=1.05, top=0.98, bottom=0.05)
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def plot_loss(
        train_loss: list[int, float] | None = None,
        test_loss: list[int, float] | None = None,
        file_path: str = "loss.png",
    ):
        assert (
            train_loss or test_loss
        ), "Either `train_loss` or `test_loss` must be provided"
        assert (
            train_loss or len(train_loss[0]) == 2
        ), "`train_loss` must be a list of tuples with epoch and loss"
        assert (
            test_loss or len(test_loss[0]) == 2
        ), "`test_loss` must be a list of tuples with epoch and loss"

        epochs = [x[0] for x in train_loss] if train_loss else [x[0] for x in test_loss]
        train_values = [x[1] for x in train_loss] if train_loss else []
        test_values = [x[1] for x in test_loss] if test_loss else []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if train_loss:
            ax.plot(epochs, train_values, label="Train")
        if test_loss:
            ax.plot(epochs, test_values, label="Test")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        if train_loss and test_loss:
            ax.legend()
        fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.05)
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def plot_confidences(
        confidences: list[float], file_path: str = "confidences.png", axis: bool = True
    ):
        assert isinstance(
            confidences[0], float
        ), "`confidences` must be a list of floats"

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(confidences)
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Confidence")
        fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.05)
        if not axis:
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close(fig)
            return data
        else:
            fig.savefig(file_path)
            plt.close(fig)

    @staticmethod
    def plot_action_segmentation(
        pred: list[str] | np.ndarray | Tensor | None = None,
        gt: list[str] | np.ndarray | Tensor | None = None,
        confidences: list[float] | None = None,
        file_path: str = "action_segmentation.png",
        mapping: dict[str, int] | None = dict(),
        backgrounds: list[str] | np.ndarray | Tensor = [],
        num_classes: int = 50,
        palette: list[tuple[float]] | None = None,
        axis: bool = True,
    ):
        assert (
            pred is not None or gt is not None
        ), "Either `pred` or `gt` must be provided"

        if pred is not None and gt is None:
            _pred, mapping = Visualizer.to_np(pred, mapping)
            pred_segments, mapping = Visualizer.to_segments(_pred, backgrounds, mapping)
            max_len = len(pred_segments)
            acc = [0]
            fig_size = (10, 2)
        elif pred is None and gt is not None:
            _gt, mapping = Visualizer.to_np(gt, mapping)
            gt_segments, mapping = Visualizer.to_segments(_gt, backgrounds, mapping)
            max_len = len(gt_segments)
            acc = [0]
            fig_size = (10, 2)
        elif pred is not None and gt is not None:
            _pred, mapping = Visualizer.to_np(pred, mapping)
            _gt, mapping = Visualizer.to_np(gt, mapping)
            pred_segments, mapping = Visualizer.to_segments(_pred, backgrounds, mapping)
            gt_segments, mapping = Visualizer.to_segments(_gt, backgrounds, mapping)
            max_len = max(len(pred_segments), len(gt_segments))
            pred_segments += [(0, (0, 0))] * (max_len - len(pred_segments))
            gt_segments += [(0, (0, 0))] * (max_len - len(gt_segments))
            acc = [0, 0]
            fig_size = (10, 3)

        if confidences is not None:
            _confidences = Visualizer.to_np(confidences)[0]
            fig_size = (10, 4)

        fig = plt.figure(figsize=fig_size)
        if pred is None or gt is None and confidences is None:
            bar_ax: Axes = fig.add_axes([0.03, 0.3, 0.94, 0.6])
        elif pred is not None and gt is not None and confidences is None:
            bar_ax: Axes = fig.add_axes([0.05, 0.15, 0.94, 0.8])
        else:
            bar_ax: Axes = fig.add_axes([0.06, 0.15 + 0.8 / 3, 0.92, 0.8 / 3 * 2])

        if palette is None:
            palette = template(num_classes, "cividis")

        for i in range(max_len):
            if pred is not None and gt is None:
                bar_ax.barh(
                    "pred",
                    pred_segments[i][1][1] - pred_segments[i][1][0],
                    color=palette[pred_segments[i][0]],
                    left=acc,
                )
                acc[0] += pred_segments[i][1][1] - pred_segments[i][1][0]
            elif pred is None and gt is not None:
                bar_ax.barh(
                    "GT",
                    gt_segments[i][1][1] - gt_segments[i][1][0],
                    color=palette[gt_segments[i][0]],
                    left=acc,
                )
                acc[0] += gt_segments[i][1][1] - gt_segments[i][1][0]
            elif pred is not None and gt is not None:
                bar_ax.barh(
                    ["Pred", "GT"],
                    [
                        pred_segments[i][1][1] - pred_segments[i][1][0],
                        gt_segments[i][1][1] - gt_segments[i][1][0],
                    ],
                    color=[palette[pred_segments[i][0]], palette[gt_segments[i][0]]],
                    left=acc,
                )
                acc[0] += pred_segments[i][1][1] - pred_segments[i][1][0]
                acc[1] += gt_segments[i][1][1] - gt_segments[i][1][0]
        bar_ax.set_xlim(right=acc[0])
        bar_ax.set_xlabel("Frame")
        if pred is None or gt is None:
            bar_ax.set_yticks([])

        if confidences is not None:
            line_ax: Axes = fig.add_axes([0.06, 0.15, 0.92, 0.8 / 3])
            line_ax.plot(_confidences, color="black")
            line_ax.set_ylim(bottom=0, top=1)
            line_ax.set_xmargin(0)
            line_ax.set_xlabel("Frame")
            line_ax.set_ylabel("Confidence")
            bar_ax.set_xticks([])
            bar_ax.set_xlabel("")
        if axis:
            fig.savefig(file_path)
            plt.close(fig)
        else:
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            data = cv2.cvtColor(data, cv2.COLOR_RGBA2RGB)
            plt.close(fig)
            return data

    @staticmethod
    def make_video(
        pred: list[str] | np.ndarray | Tensor | None = None,
        gt: list[str] | np.ndarray | Tensor | None = None,
        confidences: list[float] | None = None,
        image_dir: str | None = None,
        images: list[str] | list[np.ndarray] | None = None,
        video_path: str | None = None,
        file_path: str = "action_segmentation.mp4",
        backgrounds: list[str] | np.ndarray | Tensor = [],
        mapping: dict[str, int] | None = dict(),
        reverse_mapping: dict[int, str] | None = dict(),
        num_classes: int = 50,
        show_label: bool = True,
    ):
        assert (
            image_dir is not None or images is not None or video_path is not None
        ), "Either `image_dir`, `images`, or `video_path` must be provided"
        if images is None and image_dir is not None:
            images = Visualizer.load_images(image_dir)
        if images is None and video_path is not None:
            cap = cv2.VideoCapture(video_path)
            images = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                images.append(frame)
            cap.release()
        img_h, img_w, _ = Visualizer.load_image(images[0]).shape
        text_h = 80 if show_label else 0
        video_size = [img_w, img_h]

        _pred, mapping = (
            Visualizer.to_np(pred, mapping) if pred is not None else (None, mapping)
        )
        _gt, mapping = (
            Visualizer.to_np(gt, mapping) if gt is not None else (None, mapping)
        )
        seg_canvas = Visualizer.plot_action_segmentation(
            pred=_pred,
            gt=_gt,
            confidences=confidences,
            backgrounds=backgrounds,
            num_classes=num_classes,
            mapping=mapping,
            axis=False,
        )
        seg_h, seg_w, _ = seg_canvas.shape
        video_size[1] += seg_h
        seg_canvas = cv2.cvtColor(seg_canvas, cv2.COLOR_RGB2BGR)
        seg_canvas = cv2.resize(seg_canvas, None, fx=img_w / seg_w, fy=1)
        bar_width = 5
        w_scale = img_w * 0.92 / len(images)

        if show_label:
            video_size[1] += text_h

        with VideoWriter(
            filename=file_path,
            framerate=30,
            size=video_size,
        ) as writer:
            for i in tqdm(range(len(images)), leave=False):
                image = Visualizer.load_image(images[i])

                if pred is None or gt is None:
                    seg = seg_canvas.copy()
                    bar_start = int(i / len(images) * img_w * 0.94 + 55)
                    bar_end = min(bar_start + bar_width, len(images) - 1)
                    seg[:, bar_start:bar_end] = [255, 255, 255]
                    image = np.concatenate([image, seg], axis=0)
                elif pred is not None and gt is not None:
                    seg = seg_canvas.copy()
                    offset = img_w * 0.06
                    w_scale = img_w * 0.92 / len(images)
                    bar_start = int(offset + i * w_scale)
                    bar_end = int(offset + (i + 1) * w_scale)
                    seg[:, bar_start:bar_end] = [255, 255, 255]
                    image = np.concatenate([image, seg], axis=0)
                if show_label:
                    text_bar = np.full((text_h, img_w, 3), 255, dtype=np.uint8)
                    if pred is not None and gt is None:
                        cv2.putText(
                            img=text_bar,
                            text="pred: " + reverse_mapping[_pred[i]],
                            org=(img_w // 30, text_h // 4 * 3),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 0, 0),
                            thickness=5,
                        )
                    if pred is None and gt is not None:
                        cv2.putText(
                            img=text_bar,
                            text="gt: " + reverse_mapping[_gt[i]],
                            org=(img_w // 30, text_h // 4 * 3),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 0, 0),
                            thickness=5,
                        )
                    if pred is not None and gt is not None:
                        cv2.putText(
                            img=text_bar,
                            text="pred: " + reverse_mapping[_pred[i]],
                            org=(img_w // 30, text_h // 4 * 3),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 0, 0),
                            thickness=5,
                        )
                        cv2.putText(
                            img=text_bar,
                            text="gt: " + reverse_mapping[_gt[i]],
                            org=(img_w * 16 // 30, text_h // 4 * 3),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 0, 0),
                            thickness=5,
                        )
                    image = np.concatenate([image, text_bar], axis=0)

                writer.update(image)

    def init(
        self,
        file_path: str = None,
        mapping_path: str = None,
        has_header: bool = False,
        palette: list = None,
    ):
        file = Path(file_path)
        if file.suffix == ".mp4":
            self.writer = VideoWriter(filename=file.stem + "_tmp.mp4")
        else:
            self.file_path = file_path

        if mapping_path is not None:
            self.mapping, self.int_to_text = get_mapping(mapping_path, has_header)

        if palette is not None:
            self.palette = palette

    def close(self):
        self.writer.close()

    def segment(
        self,
        pred: list[str] | np.ndarray | Tensor,
        gt: list[str] | np.ndarray | Tensor = None,
        confidences: list[float] = None,
        file_path: str | None = "action_segmentation.png",
    ):
        output_path = file_path if self.file_path is None else self.file_path
        Visualizer.plot_action_segmentation(
            pred=pred,
            gt=gt,
            confidences=confidences,
            file_path=output_path,
            num_classes=self.num_classes,
            backgrounds=self.backgrounds,
            mapping=self.mapping if self.mapping is not None else dict(),
            palette=self.palette,
        )

    def add(self, image: np.ndarray):
        if self.writer is None:
            print(
                "[WARNING] VideoWriter is not initialized. Initializing with default filename 'action_segmentation.mp4'"
            )
            self.init("action_segmentation.mp4")
        self.writer.update(image)

    def video(
        self,
        pred: list[str] | np.ndarray | Tensor | None = None,
        gt: list[str] | np.ndarray | Tensor = None,
        confidences: list[float] = None,
    ):
        mapping = dict() if self.mapping is None else self.mapping
        if pred is not None:
            self.pred, mapping = Visualizer.to_np(pred, mapping)
        if gt is not None:
            self.gt, mapping = Visualizer.to_np(gt, mapping)
        if confidences is not None:
            self.confidences = Visualizer.to_np(confidences)[0]

        self.writer.close()
        video_path = self.writer.filename.replace("_tmp.mp4", ".mp4")
        Visualizer.make_video(
            pred=self.pred,
            gt=self.gt,
            confidences=self.confidences,
            file_path=video_path,
            video_path=self.writer.filename,
            backgrounds=self.backgrounds,
            num_classes=self.num_classes,
            mapping=mapping,
            reverse_mapping=self.int_to_text,
        )
        os.remove(self.writer.filename)
