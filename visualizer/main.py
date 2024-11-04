from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, RocCurveDisplay

from base import Base
from visualizer.palette import template
from visualizer.writer import VideoWriter
from visualizer.reader import VideoReader

from torch import Tensor
from numpy import ndarray

matplotlib.use("Agg")


class Visualizer(Base):
    metrics: dict[str, list[tuple[int, float]]] = dict()

    def __init__(self):
        super().__init__(name="Visualizer")

    @staticmethod
    def plot_feature(feature: ndarray, file_path: str = "feature.png"):
        assert isinstance(feature, ndarray) or isinstance(
            feature, Tensor
        ), "Feature must be a numpy array or a torch tensor"
        assert len(feature.shape) == 2, "Feature must be a 2D array"

        fig = plt.figure()
        ax = fig.add_subplot(111)
        axfig = ax.imshow(feature, aspect="auto", interpolation="none", cmap="jet")
        fig.colorbar(axfig, ax=ax)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.05)
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def plot_tsne(feature: ndarray, file_path: str = "tsne.png"):
        """
        feature: 2D array of shape (n_frames, n_features)
        """
        assert isinstance(feature, ndarray) or isinstance(
            feature, Tensor
        ), "Feature must be a numpy array or a torch tensor"
        assert len(feature.shape) == 2, "Feature must be a 2D array"

        tsne = TSNE(n_components=2)
        results = tsne.fit_transform(feature)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        axfig = ax.scatter(results[:, 0], results[:, 1])
        fig.colorbar(axfig, ax=ax)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.05)
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def plot_loss(
        train_loss: list[tuple[int, float]] | None = None,
        test_loss: list[tuple[int, float]] | None = None,
        file_path: str = "loss.png",
    ):
        assert (
            train_loss or test_loss
        ), "Either `train_loss` or `test_loss` must be provided"
        assert (
            train_loss and len(train_loss[0]) == 2
        ), "`train_loss` must be a list of tuples with epoch and loss"
        assert (
            test_loss and len(test_loss[0]) == 2
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
    def plot_metrics(
        metrics_name: str,
        x: list[int],
        values: list[float],
        file_path: str = "metrics.png",
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, values)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metrics_name)
        fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.05)
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def plot_confidences(
        confidences: ndarray, file_path: str = "confidences.png", axis: bool = True
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(confidences)
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Confidence")
        fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.05)
        if not axis:
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            data = np.array(canvas.buffer_rgba(), dtype="uint8")[:, :, :3]
            plt.close(fig)
            return data
        else:
            fig.savefig(file_path)
            plt.close(fig)

    @staticmethod
    def plot_action_segmentation(
        pred: ndarray | None = None,
        gt: ndarray | None = None,
        confidences: ndarray | None = None,
        file_path: Path | str = "action_segmentation.png",
        backgrounds: np.ndarray = np.array([]),
        num_classes: int = 50,
        int_to_text: dict[int, str] = dict(),
        palette: list[tuple[float, float, float, float]] | None = None,
        return_canvas: bool = False,
    ):
        assert (
            pred is not None or gt is not None
        ), "Either `pred` or `gt` must be provided"

        if pred is not None and gt is None:
            _pred = Visualizer.mask_label_with_backgrounds(pred, backgrounds)
            pred_segments = Visualizer.to_segments(_pred)
            unique_label = Visualizer.unique([x[0] for x in pred_segments])
            max_num_segments = len(pred_segments)
            fig_size = (10, 3)
        elif pred is None and gt is not None:
            _gt = Visualizer.mask_label_with_backgrounds(gt, backgrounds)
            gt_segments = Visualizer.to_segments(_gt)
            unique_label = Visualizer.unique([x[0] for x in gt_segments])
            max_num_segments = len(gt_segments)
            fig_size = (10, 3)
        elif pred is not None and gt is not None:
            _pred = Visualizer.mask_label_with_backgrounds(pred, backgrounds)
            pred_segments = Visualizer.to_segments(_pred)
            _gt = Visualizer.mask_label_with_backgrounds(gt, backgrounds)
            gt_segments = Visualizer.to_segments(_gt)
            unique_label = Visualizer.unique(
                [x[0] for x in gt_segments] + [x[0] for x in pred_segments]
            )
            max_num_segments = max(len(pred_segments), len(gt_segments))
            pred_segments += [(-1, (0, 0))] * (max_num_segments - len(pred_segments))
            gt_segments += [(-1, (0, 0))] * (max_num_segments - len(gt_segments))
            fig_size = (10, 4)

        if confidences is not None:
            fig_size = (10, fig_size[1] + 1)

        fig = plt.figure(figsize=fig_size)
        gs = fig.add_gridspec(3, 1)
        acc = [0, 0]  # [Pred, GT]

        if (pred is None or gt is None) and confidences is None:
            fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.5)
            legend_anchor = (0.5, -0.5)
            bar_ax = fig.add_subplot(gs[:, :])  # type: ignore
        elif (pred is None or gt is None) and confidences is not None:
            bar_ax = fig.add_subplot(gs[0:3])
            legend_anchor = (0.5, -0.5)
        elif (pred is not None and gt is not None) and confidences is None:
            bar_ax = fig.add_subplot(gs[0:3])
            legend_anchor = (0.5, -0.5)
        elif (pred is not None and gt is not None) and confidences is not None:
            fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.2)
            bar_ax = fig.add_subplot(gs[0:2])
            legend_anchor = (0.4, -0.7)

        if palette is None:
            palette = template(num_classes, "cividis")

        target_bar = []
        for i in range(max_num_segments):
            if pred is not None and gt is None:
                p = bar_ax.barh(
                    "Pred",
                    pred_segments[i][1][1] - pred_segments[i][1][0],
                    color=palette[pred_segments[i][0]],
                    left=acc,
                )
                acc[0] += pred_segments[i][1][1] - pred_segments[i][1][0]
                if pred_segments[i][0] not in [x[0] for x in target_bar]:
                    target_bar.append((pred_segments[i][0], p[0]))
            elif pred is None and gt is not None:
                p = bar_ax.barh(
                    "GT",
                    gt_segments[i][1][1] - gt_segments[i][1][0],
                    color=palette[gt_segments[i][0]],
                    left=acc,
                )
                acc[0] += int(gt_segments[i][1][1] - gt_segments[i][1][0])
                if gt_segments[i][0] not in [x[0] for x in target_bar]:
                    target_bar.append((gt_segments[i][0], p[0]))
            elif pred is not None and gt is not None:
                p = bar_ax.barh(
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
                if pred_segments[i][0] not in [x[0] for x in target_bar]:
                    target_bar.append((pred_segments[i][0], p[0]))
                if gt_segments[i][0] not in [x[0] for x in target_bar]:
                    target_bar.append((gt_segments[i][0], p[1]))

        bar_ax.set_xlim(right=acc[0])
        bar_ax.set_xlabel("Frame")
        ordered_target_bar = sum([target_bar[i::7] for i in range(7)], [])
        ordered_unique_label = np.concatenate([unique_label[i::7] for i in range(7)])
        bar_ax.legend(
            [x[1] for x in ordered_target_bar],
            [
                int_to_text[int(ordered_unique_label[i])]
                for i in range(len(ordered_unique_label))
            ],
            bbox_to_anchor=legend_anchor,
            loc="upper center",
            ncols=7,
        )
        if pred is None or gt is None:
            bar_ax.set_yticks([])

        if confidences is not None:
            line_ax = fig.add_subplot(gs[2])
            line_ax.plot(confidences, color="black")
            line_ax.set_ylim(bottom=0, top=1)
            line_ax.set_xmargin(0)
            line_ax.set_xlabel("Frame")
            line_ax.set_ylabel("Confidence")
            bar_ax.set_xticks([])
            bar_ax.set_xlabel("")
        if return_canvas:
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            data = np.array(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
            plt.close(fig)
            return data
        else:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(file_path)
            plt.close(fig)

    @staticmethod
    def plot_roc_curve(
        gt: ndarray,
        pred: ndarray,
        file_path: str = "roc_curve.png",
    ):
        fpr, tpr, _ = roc_curve(gt, pred)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
        fig, ax = plt.subplots()
        roc_display.plot(ax=ax)
        fig.savefig(file_path)
        plt.close(fig)

    @staticmethod
    def make_video(
        pred: ndarray | None = None,
        gt: ndarray | None = None,
        confidences: ndarray | None = None,
        image_dir: str | None = None,
        images: list[str] | list[np.ndarray] | None = None,
        video_path: str | None = None,
        out_path: str = "action_segmentation.mp4",
        backgrounds: ndarray = np.array([]),
        mapping: dict[str, int] = dict(),
        reverse_mapping: dict[int, str] = dict(),
        num_classes: int = 50,
        show_segment: bool = True,
        show_label: bool = True,
    ):
        reader = VideoReader(image_dir=image_dir, images=images, video_path=video_path)
        video_size = reader.image_size
        text_area_size = (80, video_size[1])

        if show_segment:
            segmentation = Visualizer.plot_action_segmentation(
                pred=pred,
                gt=gt,
                confidences=confidences,
                backgrounds=backgrounds,
                num_classes=num_classes,
                return_canvas=True,
            )
            segmentation = cv2.resize(
                segmentation,  # type: ignore
                None,
                fx=video_size[1] / segmentation.shape[1],  # type: ignore
                fy=1.0,
            )  # type: ignore
            video_size = (video_size[0] + segmentation.shape[0], video_size[1])  # type: ignore
            bar_width = max(5, int(segmentation.shape[1] * 0.005))  # type: ignore

        if show_label:
            video_size = (video_size[0] + text_area_size[0], video_size[1])

        with VideoWriter(
            filename=out_path,
            framerate=30,
            size=video_size,
        ) as writer:
            for i, image in enumerate(tqdm(reader, total=reader.num_frames)):
                if show_segment:
                    seg = segmentation.copy()  # type: ignore
                    bar_start = i
                    bar_end = i + bar_width
                    seg[:, bar_start:bar_end] = 250
                    image = np.concatenate([image, seg], axis=0)
                if show_label:
                    seg = np.full(
                        (text_area_size[0], image.shape[1], 3), 255, dtype=np.uint8
                    )
                    if pred is not None and gt is None:
                        cv2.putText(
                            img=seg,  # type: ignore
                            text="Pred: " + reverse_mapping[pred[i]],
                            org=(
                                reader.image_size[1] // 30,
                                reader.image_size[0] // 4 * 3,
                            ),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 0, 0),
                            thickness=5,
                        )
                    if pred is None and gt is not None:
                        cv2.putText(
                            img=seg,  # type: ignore
                            text="GT: " + reverse_mapping[gt[i]],
                            org=(
                                reader.image_size[1] // 30,
                                reader.image_size[0] // 4 * 3,
                            ),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 0, 0),
                            thickness=5,
                        )
                    if pred is not None and gt is not None:
                        cv2.putText(
                            img=seg,  # type: ignore
                            text="Pred: " + reverse_mapping[pred[i]],
                            org=(
                                reader.image_size[1] // 30,
                                reader.image_size[0] // 4 * 3,
                            ),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 0, 0),
                            thickness=5,
                        )
                        cv2.putText(
                            img=seg,  # type: ignore
                            text="GT: " + reverse_mapping[gt[i]],
                            org=(
                                reader.image_size[1] * 16 // 30,
                                reader.image_size[0] // 4 * 3,
                            ),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 0, 0),
                            thickness=5,
                        )
                    image = np.concatenate([image, seg], axis=0)

                writer.update(image)

    def init(self):
        pass

    def add_metrics(self, epoch: int, metrics: dict[str, float]):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = [(epoch, value)]
            else:
                self.metrics[key].append((epoch, value))

    def save_metrics(self, save_dir: str):
        for key, values in self.metrics.items():
            epochs = [x[0] for x in values]
            metrics = [x[1] for x in values]
            self.plot_metrics(key, epochs, metrics, f"{save_dir}/{key}.png")
