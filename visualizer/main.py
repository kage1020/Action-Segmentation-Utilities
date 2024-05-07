import os
import sys
import glob
import pathlib
from tqdm import tqdm
import matplotlib.figure as fgr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import ffmpeg
import numpy as np
import torch
from writer import VideoWriter


class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def to_np(x: list | np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, list):
            if len(x) > 0 and isinstance(x[0], str):
                return np.array(Visualizer.str_to_int(x)[0])
            return np.array(x)
        raise ValueError("Invalid input type")

    @staticmethod
    def to_segments(
        x: list | np.ndarray | torch.Tensor,
        backgrounds: list | np.ndarray | torch.Tensor
    ) -> list[int, tuple[int, int]]:
        _x = Visualizer.to_np(x)
        _backgrounds = Visualizer.to_np(backgrounds)
        diff = np.diff(_x, prepend=-1)
        indices = np.where(diff != 0)[0]
        segments = []

        for i in range(len(indices)):
            if x[indices[i]] in _backgrounds:
                continue
            start = indices[i]
            end = indices[i + 1] if i + 1 < len(indices) else len(x)
            segments.append((x[start], (start, end)))

        return segments

    @staticmethod
    def str_to_int(x: list[str]) -> list[int]:
        mapping = dict()
        for i, s in enumerate(x):
            if s not in mapping:
                mapping[s] = len(mapping)
        return [mapping[s] for s in x], mapping

    @staticmethod
    def plot_feature(feature, file_path: str):
        assert isinstance(feature, np.ndarray) or isinstance(feature, torch.Tensor), 'Feature must be a numpy array or a torch tensor'
        assert len(feature.shape) == 2, 'Feature must be a 2D array'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        axfig = ax.imshow(feature, aspect='auto', interpolation='none', cmap='jet')
        fig.colorbar(axfig, ax=ax)
        fig.subplots_adjust(left=0.1, right=1.05, top=0.98, bottom=0.05)
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def plot_loss(losses: list[int, float], file_path: str):
        assert len(losses[0]) == 2, '`Losses` must be a list of tuples with epoch and loss'

        epochs = [x[0] for x in losses]
        values = [x[1] for x in losses]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epochs, values)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.05)
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def plot_train_val_loss(losses: list[int, float, float], file_path: str):
        assert len(losses[0]) == 3, '`losses` must be a list of tuples with epoch, train loss and validation loss'

        epochs = [x[0] for x in losses]
        train_values = [x[1] for x in losses]
        val_values = [x[2] for x in losses]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epochs, train_values, label='Train')
        ax.plot(epochs, val_values, label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.05)
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def plot_confidences_without_axis(confidences: list[float]):
        assert isinstance(confidences[0], float), '`confidences` must be a list of floats'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(confidences)
        fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.05)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return data

    @staticmethod
    def plot_confidences(confidences: list[float], file_path: str):
        assert isinstance(confidences[0], float), '`confidences` must be a list of floats'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(confidences)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Confidence')
        fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.05)
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def plot_action_segmentation_without_axis(
        x: list[str] | np.ndarray | torch.Tensor,
        backgrounds: list[str] | np.ndarray | torch.Tensor = [],
        num_classes: int = 50
    ):
        _x = Visualizer.to_np(x)
        x_segments = Visualizer.to_segments(_x, backgrounds)
        acc = 0

        fig = plt.figure(figsize=(10, 1))
        ax = fig.add_subplot(111)
        ax.axis('off')
        for s in x_segments:
            color = cm.jet(s[0] / (num_classes + 1))
            ax.barh(0, s[1][1] - s[1][0], color=color, left=acc)
            acc += s[1][1] - s[1][0]
        ax.set_xlim(right=acc)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return data

    @staticmethod
    def plot_action_segmentation_with_gt_without_x_axis(
        pred: list[str] | np.ndarray | torch.Tensor,
        gt: list[str] | np.ndarray | torch.Tensor,
        backgrounds: list[str] | np.ndarray | torch.Tensor = [],
        num_classes: int = 50
    ):
        _pred = Visualizer.to_np(pred)
        _gt = Visualizer.to_np(gt)
        pred_segments = Visualizer.to_segments(_pred, backgrounds)
        gt_segments = Visualizer.to_segments(_gt, backgrounds)
        max_len = max(len(pred_segments), len(gt_segments))
        pred_segments += [(0, (0, 0))] * (max_len - len(pred_segments))
        gt_segments += [(0, (0, 0))] * (max_len - len(gt_segments))
        acc = [0, 0]

        fig = plt.figure(figsize=(10, 2))
        ax = fig.add_subplot(111)
        for i in range(max_len):
            pred_color = cm.jet(pred_segments[i][0] / (num_classes + 1))
            gt_color = cm.jet(gt_segments[i][0] / (num_classes + 1))
            ax.barh(['Pred', 'GT'], [pred_segments[i][1][1] - pred_segments[i][1][0], gt_segments[i][1][1] - gt_segments[i][1][0]], color=[gt_color, pred_color], left=acc)
            acc[0] += pred_segments[i][1][1] - pred_segments[i][1][0]
            acc[1] += gt_segments[i][1][1] - gt_segments[i][1][0]
        ax.set_xlim(right=acc[1])
        ax.set_xlabel('Frame')
        fig.subplots_adjust(left=0.05, right=1, top=1, bottom=0)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return data

    @staticmethod
    def plot_action_segmentation(
        x: list[str] | np.ndarray | torch.Tensor,
        file_path: str,
        backgrounds: list[str] | np.ndarray | torch.Tensor = [],
        num_classes: int = 50
    ):
        _x = Visualizer.to_np(x)
        x_segments = Visualizer.to_segments(_x, backgrounds)
        acc = 0

        fig = plt.figure(figsize=(10, 1))
        ax = fig.add_subplot(111)
        for s in x_segments:
            color = cm.jet(s[0] / (num_classes + 1))
            ax.barh(0, s[1][1] - s[1][0], color=color, left=acc)
            acc += s[1][1] - s[1][0]
        ax.set_xlim(right=acc)
        ax.set_yticks([])
        ax.set_xlabel('Frame')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.3)
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def plot_action_segmentation_with_gt(
        pred: list[str] | np.ndarray | torch.Tensor,
        gt: list[str] | np.ndarray | torch.Tensor,
        file_path: str,
        backgrounds: list[str] | np.ndarray | torch.Tensor = [],
        num_classes: int = 50
    ):
        _pred = Visualizer.to_np(pred)
        _gt = Visualizer.to_np(gt)
        pred_segments = Visualizer.to_segments(_pred, backgrounds)
        gt_segments = Visualizer.to_segments(_gt, backgrounds)
        max_len = max(len(pred_segments), len(gt_segments))
        pred_segments += [(0, (0, 0))] * (max_len - len(pred_segments))
        gt_segments += [(0, (0, 0))] * (max_len - len(gt_segments))
        acc = [0, 0] # pred, gt

        fig = plt.figure(figsize=(10, 2))
        ax = fig.add_subplot(111)
        for i in range(max_len):
            pred_color = cm.jet(pred_segments[i][0] / (num_classes + 1))
            gt_color = cm.jet(gt_segments[i][0] / (num_classes + 1))
            ax.barh(['Pred', 'GT'], [pred_segments[i][1][1] - pred_segments[i][1][0], gt_segments[i][1][1] - gt_segments[i][1][0]], color=[gt_color, pred_color], left=acc)
            acc[0] += pred_segments[i][1][1] - pred_segments[i][1][0]
            acc[1] += gt_segments[i][1][1] - gt_segments[i][1][0]
        ax.set_xlim(right=acc[1])
        ax.set_xlabel('Frame')
        fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.15)
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def plot_action_segmentation_with_confidence(
        x: list[str] | np.ndarray | torch.Tensor,
        confidences: list[float],
        file_path: str,
        backgrounds: list[str] | np.ndarray | torch.Tensor = [],
        num_classes: int = 50
    ):
        _x = Visualizer.to_np(x)
        x_segments = Visualizer.to_segments(_x, backgrounds)
        acc = 0
        _confidences = Visualizer.to_np(confidences)

        fig = plt.figure(figsize=(10, 2))
        bar_ax = fig.add_axes([0.06, 0.15+0.8/3, 0.92, 0.8/3*2])
        for s in x_segments:
            color = cm.jet(s[0] / (num_classes + 1))
            bar_ax.barh(0, s[1][1] - s[1][0], color=color, left=acc)
            acc += s[1][1] - s[1][0]
        bar_ax.set_xlim(right=acc)
        bar_ax.set_xticks([])
        bar_ax.set_yticks([])

        line_ax = fig.add_axes([0.06, 0.15, 0.92, 0.8/3-0.05])
        line_ax.plot(range(1, len(_confidences)+1), _confidences, color='black')
        line_ax.set_ylim(bottom=0, top=1)
        line_ax.set_xmargin(0)
        line_ax.set_xlabel('Frame')
        line_ax.set_ylabel('Confidence')
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def plot_action_segmentation_with_gt_confidence(
        pred: list[str] | np.ndarray | torch.Tensor,
        gt: list[str] | np.ndarray | torch.Tensor,
        confidences: list[float],
        file_path: str,
        backgrounds: list[str] | np.ndarray | torch.Tensor = [],
        num_classes: int = 50
    ):
        _pred = Visualizer.to_np(pred)
        _gt = Visualizer.to_np(gt)
        pred_segments = Visualizer.to_segments(_pred, backgrounds)
        gt_segments = Visualizer.to_segments(_gt, backgrounds)
        max_len = max(np.max(_pred), np.max(_gt))
        pred_segments += [(0, (0, 0))] * (max_len - len(pred_segments))
        gt_segments += [(0, (0, 0))] * (max_len - len(gt_segments))
        acc = [0, 0]
        _confidences = Visualizer.to_np(confidences)

        fig = plt.figure(figsize=(10, 3))
        bar_ax = fig.add_axes([0.06, 0.15+0.8/3, 0.92, 0.8/3*2])
        for i in range(max_len):
            pred_color = cm.jet(pred_segments[i][0] / (num_classes + 1))
            gt_color = cm.jet(gt_segments[i][0] / (num_classes + 1))
            bar_ax.barh(['Pred', 'GT'], [pred_segments[i][1][1] - pred_segments[i][1][0], gt_segments[i][1][1] - gt_segments[i][1][0]], color=[gt_color, pred_color], left=acc)
            acc[0] += pred_segments[i][1][1] - pred_segments[i][1][0]
            acc[1] += gt_segments[i][1][1] - gt_segments[i][1][0]
        bar_ax.set_xlim(right=acc[1])
        bar_ax.set_xticks([])

        line_ax = fig.add_axes([0.06, 0.15, 0.92, 0.8/3-0.05])
        line_ax.plot(range(1, len(_confidences)+1), _confidences, color='black')
        line_ax.set_ylim(bottom=0, top=1)
        line_ax.set_xmargin(0)
        line_ax.set_xlabel('Frame')
        line_ax.set_ylabel('Confidence')
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def make_action_segmentation_video(
        pred: list[str] | np.ndarray | torch.Tensor,
        image_dir: str,
        file_path: str,
        backgrounds: list[str] | np.ndarray | torch.Tensor = [],
        num_classes: int = 50
    ):
        _pred = Visualizer.to_np(pred)
        image_paths = glob.glob(f'{image_dir}/*.png')
        image_paths += glob.glob(f'{image_dir}/*.jpg')
        image_paths.sort()
        height, width, _ = cv2.imread(image_paths[0]).shape
        fps = 30
        text_height = 100

        canvas = Visualizer.plot_action_segmentation_without_axis(_pred, backgrounds=backgrounds, num_classes=num_classes)
        h, _, _ = canvas.shape
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGBA2RGB)
        canvas = cv2.resize(canvas, (width, h))

        with VideoWriter(
            filename=file_path,
            framerate=fps,
            size=(width, height + h + text_height),
        ) as writer:
            for i in tqdm(range(len(_pred)), leave=False):
                image = cv2.imread(image_paths[i])
                seg = canvas.copy()
                bar_start = max(i - 1, 0)
                bar_end = min(i + 1, len(_pred) - 1)
                seg[:, bar_start:bar_end] = [255, 255, 255]
                image = np.concatenate([image, seg], axis=0)
                text_bar = np.full((text_height, width, 3), 255, dtype=np.uint8)
                cv2.putText(
                    img=text_bar,
                    text="pred: "+str(_pred[i]),
                    org=(width//2, text_height//4*3),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 0, 0),
                    thickness=5
                )
                writer.update(image)

    @staticmethod
    def make_action_segmentation_video_with_confidence(
        pred: list[str] | np.ndarray | torch.Tensor,
        confidences: list[float],
        image_dir: str,
        file_path: str,
        backgrounds: list[str] | np.ndarray | torch.Tensor = [],
        num_classes: int = 50
    ):
        _pred = Visualizer.to_np(pred)
        image_paths = glob.glob(f'{image_dir}/*.png')
        image_paths += glob.glob(f'{image_dir}/*.jpg')
        image_paths.sort()
        height, width, _ = cv2.imread(image_paths[0]).shape
        fps = 30
        text_height = 100

        seg_canvas = Visualizer.plot_action_segmentation_without_axis(_pred, backgrounds=backgrounds, num_classes=num_classes)
        seg_h, _, _ = seg_canvas.shape
        seg_canvas = cv2.cvtColor(seg_canvas, cv2.COLOR_RGBA2RGB)
        seg_canvas = cv2.resize(seg_canvas, (width, seg_h))

        conf_canvas = Visualizer.plot_confidences_without_axis(confidences)
        cf_h, _, _ = conf_canvas.shape
        conf_canvas = cv2.cvtColor(conf_canvas, cv2.COLOR_RGBA2RGB)
        conf_canvas = cv2.resize(conf_canvas, (width, cf_h))

        with VideoWriter(
            filename=file_path,
            framerate=fps,
            size=(width, height + seg_h + cf_h + text_height),
        ) as writer:
            for i in tqdm(range(len(_pred)), leave=False):
                image = cv2.imread(image_paths[i])
                seg = seg_canvas.copy()
                bar_start = max(i - 1, 0)
                bar_end = min(i + 1, len(_pred) - 1)
                seg[:, bar_start:bar_end] = [255, 255, 255]
                conf = conf_canvas.copy()
                text_bar = np.full((text_height, width, 3), 255, dtype=np.uint8)
                cv2.putText(
                    img=text_bar,
                    text="pred: "+str(_pred[i]),
                    org=(width//2, text_height//4*3),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 0, 0),
                    thickness=5
                )
                image = np.concatenate([image, seg, conf, text_bar], axis=0)
                writer.update(image)

    @staticmethod
    def make_action_segmentation_video_with_gt(
        pred: list[str] | np.ndarray | torch.Tensor,
        gt: list[str] | np.ndarray | torch.Tensor,
        image_dir: str,
        file_path: str,
        backgrounds: list[str] | np.ndarray | torch.Tensor = [],
        num_classes: int = 50
    ):
        _pred = Visualizer.to_np(pred)
        _gt = Visualizer.to_np(gt)
        image_paths = glob.glob(f'{image_dir}/*.png')
        image_paths += glob.glob(f'{image_dir}/*.jpg')
        image_paths.sort()
        height, width, _ = cv2.imread(image_paths[0]).shape
        fps = 30

        canvas = Visualizer.plot_action_segmentation_with_gt_without_x_axis(_pred, _gt, backgrounds=backgrounds, num_classes=num_classes)
        h, _, _ = canvas.shape
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGBA2RGB)
        canvas = cv2.resize(canvas, (width, h))
        label_ratio = 0.051
        w_scale = width * (1 - label_ratio) / len(_pred)
        offset = int(label_ratio * width)

        text_height = 100

        with VideoWriter(
            filename=file_path,
            framerate=fps,
            size=(width, height+h+text_height),
        ) as writer:
            for i in tqdm(range(len(_pred)), leave=False):
                image = cv2.imread(image_paths[i])
                seg = canvas.copy()
                bar_start = offset + int(i * w_scale)
                bar_end = offset + int((i + 1) * w_scale) + 1
                seg[:, bar_start:bar_end] = [255, 255, 255]
                text_bar = np.full((text_height, width, 3), 255, dtype=np.uint8)
                cv2.putText(
                    img=text_bar,
                    text="GT: "+str(_gt[i]),
                    org=(0, text_height//4*3),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 0, 0),
                    thickness=5
                )
                cv2.putText(
                    img=text_bar,
                    text="pred: "+str(_pred[i]),
                    org=(width//2, text_height//4*3),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 0, 0),
                    thickness=5
                )
                image = np.concatenate([image, seg, text_bar], axis=0)
                writer.update(image)

    @staticmethod
    def make_action_segmentation_video_with_gt_confidence(
        pred: list[str] | np.ndarray | torch.Tensor,
        gt: list[str] | np.ndarray | torch.Tensor,
        confidences: list[float],
        image_dir: str,
        file_path: str,
        backgrounds: list[str] | np.ndarray | torch.Tensor = [],
        num_classes: int = 50
    ):
        _pred = Visualizer.to_np(pred)
        _gt = Visualizer.to_np(gt)
        image_paths = glob.glob(f'{image_dir}/*.png')
        image_paths += glob.glob(f'{image_dir}/*.jpg')
        image_paths.sort()
        height, width, _ = cv2.imread(image_paths[0]).shape
        fps = 30

        seg_canvas = Visualizer.plot_action_segmentation_with_gt_without_x_axis(_pred, _gt, backgrounds=backgrounds, num_classes=num_classes)
        seg_h, _, _ = seg_canvas.shape
        seg_canvas = cv2.cvtColor(seg_canvas, cv2.COLOR_RGBA2RGB)
        seg_canvas = cv2.resize(seg_canvas, (width, seg_h))
        label_ratio = 0.051
        w_scale = width * (1 - label_ratio) / len(_pred)
        offset = int(label_ratio * width)

        conf_canvas = Visualizer.plot_confidences_without_axis(confidences)
        cf_h, _, _ = conf_canvas.shape
        conf_canvas = cv2.cvtColor(conf_canvas, cv2.COLOR_RGBA2RGB)
        conf_canvas = cv2.resize(conf_canvas, (width, cf_h))

        text_height = 100

        with VideoWriter(
            filename=file_path,
            framerate=fps,
            size=(width, height + seg_h + text_height),
        ) as writer:
            for i in tqdm(range(len(_pred)), leave=False):
                image = cv2.imread(image_paths[i])
                seg = seg_canvas.copy()
                bar_start = offset + int(i * w_scale)
                bar_end = offset + int((i + 1) * w_scale) + 1
                seg[:, bar_start:bar_end] = [255, 255, 255]
                conf = conf_canvas.copy()
                text_bar = np.full((text_height, width, 3), 255, dtype=np.uint8)
                cv2.putText(
                    img=text_bar,
                    text="GT: "+str(_gt[i]),
                    org=(0, text_height//4*3),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 0, 0),
                    thickness=5
                )
                cv2.putText(
                    img=text_bar,
                    text="pred: "+str(_pred[i]),
                    org=(width//2, text_height//4*3),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 0, 0),
                    thickness=5
                )
                image = np.concatenate([image, seg, conf, text_bar], axis=0)
                writer.update(image)

    def segment(
        self,
        pred: list[str] | np.ndarray | torch.Tensor,
        gt: list[str] | np.ndarray | torch.Tensor = [],
        confidences: list[float] = [],
        file_path: str = "action_segmentation.png",
        backgrounds: list[str] | np.ndarray | torch.Tensor = [],
        num_classes: int = 50,
    ):
        if gt == []:
            if confidences == []:
                Visualizer.plot_action_segmentation(pred, file_path, num_classes=num_classes, backgrounds=backgrounds)
            else:
                Visualizer.plot_action_segmentation_with_confidence(pred, file_path, num_classes=num_classes, backgrounds=backgrounds)
        else:
            if confidences == []:
                Visualizer.plot_action_segmentation_with_gt(pred, gt, file_path, num_classes=num_classes, backgrounds=backgrounds)
            else:
                Visualizer.plot_action_segmentation_with_gt_confidence(pred, gt, file_path, num_classes=num_classes, backgrounds=backgrounds)

    def make_video(
        pred: list[str] | np.ndarray | torch.Tensor,
        gt: list[str] | np.ndarray | torch.Tensor = [],
        confidences: list[float] = [],
        image_dir: str = "/images",
        file_path: str = "action_segmentation.mp4",
        backgrounds: list[str] | np.ndarray | torch.Tensor = [],
        num_classes: int = 50,
    ):
        if gt == []:
            if confidences == []:
                Visualizer.make_action_segmentation_video(pred, image_dir, file_path, backgrounds=backgrounds, num_classes=num_classes)
            else:
                Visualizer.make_action_segmentation_video_with_confidence(pred, confidences, image_dir, file_path, backgrounds=backgrounds, num_classes=num_classes)
        else:
            if confidences == []:
                Visualizer.make_action_segmentation_video_with_gt(pred, gt, image_dir, file_path, backgrounds=backgrounds, num_classes=num_classes)
            else:
                Visualizer.make_action_segmentation_video_with_gt(pred, gt, image_dir, file_path, backgrounds=backgrounds, num_classes=num_classes)
