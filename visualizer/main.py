import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray
from sklearn.manifold import TSNE
from sklearn.metrics import RocCurveDisplay, roc_curve
from tqdm import tqdm

from ..base import Base, mask_label_with_backgrounds, to_segments, unique
from .palette import template
from .reader import VideoReader
from .writer import VideoWriter


def plot_image(
    feature: ndarray, file_path: str = "feature.png", is_jupyter: bool = False
):
    """
    Plots an image tensor using matplotlib.

    Args:
        feature (ndarray): A 3D NumPy array representing the image.
        file_path (str, optional): File path to save the image if not displaying in Jupyter. Defaults to "feature.png".
        is_jupyter (bool, optional): Whether to display inline in a Jupyter notebook. Defaults to False. If you want to use this function in jupyter notebook, please set `%matplotlib inline` at the top of the notebook.
    Raises:
        TypeError: If `feature` is not a NumPy ndarray.
        ValueError: If `feature` is not a 3D array.
    Returns:
        None
    """
    if not isinstance(feature, np.ndarray):
        raise TypeError("`feature` must be a NumPy ndarray.")
    if feature.ndim != 3:
        raise ValueError("`feature` must be a 3D array.")

    fig, ax = plt.subplots()
    ax.imshow(feature)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.05)

    if is_jupyter:
        from IPython.display import display

        display(fig)
    else:
        fig.savefig(file_path)
        plt.close(fig)


def plot_feature(
    feature: ndarray,
    file_path: str = "feature.png",
    is_jupyter: bool = False,
    cmap: str = "viridis",
):
    """
    Plots a feature tensor using matplotlib.

    Args:
        feature (ndarray): A 2D NumPy array representing the feature.
        file_path (str, optional): File path to save the image if not displaying in Jupyter. Defaults to "feature.png".
        is_jupyter (bool, optional): Whether to display inline in a Jupyter notebook. Defaults to False. If you want to use this function in jupyter notebook, please set `%matplotlib inline` at the top of the notebook.
        cmap (str, optional): Colormap for the plot. Defaults to "viridis".
    Raises:
        TypeError: If `feature` is not a NumPy ndarray.
        ValueError: If `feature` is not a 2D array.
    Returns:
        None
    """
    if not isinstance(feature, ndarray):
        raise TypeError("`feature` must be a NumPy ndarray.")
    if feature.ndim != 2:
        raise ValueError("`feature` must be a 2D array.")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    axfig = ax.imshow(feature, aspect="auto", interpolation="none", cmap=cmap)
    fig.colorbar(axfig, ax=ax)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.05)
    if is_jupyter:
        from IPython.display import display

        display(fig)
    else:
        fig.savefig(file_path)

    plt.close(fig)


def plot_features(
    features: list[ndarray] | ndarray,
    file_paths: list[str] = [],
    is_jupyter: bool = False,
    ncols: int = None,
    nrows: int = None,
    cmap: str = "viridis",
):
    """
    Plots multiple feature tensors using matplotlib.

    Args:
        features (list[ndarray] | ndarray): A list of 2D NumPy arrays or a 3D NumPy array representing the features.
        file_paths (list[str], optional): List of file paths to save the images if not displaying in Jupyter. Defaults to [].
        is_jupyter (bool, optional): Whether to display inline in a Jupyter notebook. Defaults to False. If you want to use this function in jupyter notebook, please set `%matplotlib inline` at the top of the notebook.
        ncols (int, optional): Number of columns for the subplot grid. Defaults to None.
        nrows (int, optional): Number of rows for the subplot grid. Defaults to None.
        cmap (str, optional): Colormap for the plot. Defaults to "viridis".
    Raises:
        TypeError: If `features` is not a list or a NumPy ndarray.
        ValueError: If `features` is not a 3D array when passed as a NumPy ndarray.
    Returns:
        None
    """
    if not isinstance(features, (list, ndarray)):
        raise TypeError("`features` must be a list or a NumPy ndarray.")
    if isinstance(features, ndarray) and len(features.shape) != 3:
        raise ValueError("`features` must be a 3D array.")
    if isinstance(features, list) and len(features) == 0:
        raise ValueError("`features` list cannot be empty.")
    if isinstance(features, list) and len(file_paths) != len(features):
        raise ValueError(
            "Length of `file_paths` must match the length of `features` list."
        )
    if isinstance(features, list) and len(file_paths) == 0:
        raise ValueError("`file_paths` list cannot be empty.")

    num_features = len(features)
    nrows = math.floor(math.sqrt(num_features)) if nrows is None else nrows
    ncols = math.ceil(num_features / nrows) if ncols is None else ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 6, nrows * 5), tight_layout=True
    )
    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        axfig = ax.imshow(feature, aspect="auto", interpolation="none", cmap=cmap)
        fig.colorbar(axfig, ax=ax)

    fig.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.05, wspace=0.3)

    if is_jupyter:
        from IPython.display import display

        display(fig)
    else:
        fig.savefig(file_paths[0])

    plt.close(fig)


def plot_tsne(
    feature: ndarray,
    gt: list[int] | None = None,
    file_path: str | Path = "tsne.png",
    is_jupyter: bool = False,
    frame_index: bool = False,
):
    """
    Plots a t-SNE visualization of the feature tensor.

    Args:
        feature (ndarray): A 2D NumPy array representing the feature.
        gt (list[int], optional): A list of ground truth labels. Defaults to None.
        file_path (str | Path, optional): File path to save the image if not displaying in Jupyter. Defaults to "tsne.png".
        is_jupyter (bool, optional): Whether to display inline in a Jupyter notebook. Defaults to False. If you want to use this function in jupyter notebook, please set `%matplotlib inline` at the top of the notebook.
        frame_index (bool, optional): Whether to plot the frame index. Defaults to False.
    Raises:
        TypeError: If `feature` is not a NumPy ndarray.
        ValueError: If `feature` is not a 2D array.
    Returns:
        None
    """
    if not isinstance(feature, ndarray):
        raise TypeError("`feature` must be a NumPy ndarray.")
    if feature.ndim != 2:
        raise ValueError("`feature` must be a 2D array.")
    if gt is not None and len(feature) != len(gt):
        raise ValueError("Length of `feature` and `gt` must be the same.")

    results = TSNE(n_components=2, random_state=0).fit_transform(feature)

    def _plot(points, colors, suffix=""):
        fig, ax = plt.subplots()
        scatter = ax.scatter(points[:, 0], points[:, 1], c=colors)
        fig.colorbar(scatter, ax=ax)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.05)

        if is_jupyter:
            from IPython.display import display

            display(fig)
        else:
            out_path = Path(file_path).with_name(
                Path(file_path).stem + suffix + Path(file_path).suffix
            )
            fig.savefig(out_path)
        plt.close(fig)

    if frame_index:
        _plot(results, range(len(gt)), "_frame_index")
    else:
        _plot(results, gt)


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
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        data = np.array(canvas.buffer_rgba(), dtype="uint8")[:, :, :3]
        plt.close(fig)
        return data
    else:
        fig.savefig(file_path)

    plt.close(fig)


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
    is_jupyter: bool = False,
    dynamic_palette: bool = False,
    legend_ncols: int = 7,
):
    assert pred is not None or gt is not None, "Either `pred` or `gt` must be provided"

    fig_size = (10, 3)
    unique_label = []
    max_num_segments = 0
    pred_segments = []
    gt_segments = []
    if pred is not None and gt is None:
        _pred = mask_label_with_backgrounds(pred, backgrounds)
        pred_segments = to_segments(_pred)
        unique_label = unique([x[0] for x in pred_segments])
        max_num_segments = len(pred_segments)
        fig_size = (10, 3)
    elif pred is None and gt is not None:
        _gt = mask_label_with_backgrounds(gt, backgrounds)
        gt_segments = to_segments(_gt)
        unique_label = unique([x[0] for x in gt_segments])
        max_num_segments = len(gt_segments)
        fig_size = (10, 3)
    elif pred is not None and gt is not None:
        _pred = mask_label_with_backgrounds(pred, backgrounds)
        pred_segments = to_segments(_pred)
        _gt = mask_label_with_backgrounds(gt, backgrounds)
        gt_segments = to_segments(_gt)
        unique_label = unique(
            [x[0] for x in gt_segments] + [x[0] for x in pred_segments]
        )
        max_num_segments = max(len(pred_segments), len(gt_segments))
        pred_segments += [(-1, (0, 0))] * (max_num_segments - len(pred_segments))
        gt_segments += [(-1, (0, 0))] * (max_num_segments - len(gt_segments))
        fig_size = (10, 4)

    if confidences is not None:
        fig_size = (10, fig_size[1] + 1)

    num_legend_rows = math.ceil(len(unique_label) / legend_ncols)
    if num_legend_rows > 3:
        fig_size = (fig_size[0], fig_size[1] + (num_legend_rows - 3))

    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(3, 1)
    acc = [0, 0]  # [Pred, GT]

    bar_ax: Axes = None
    legend_anchor = (0.5, -0.5)
    if (pred is None or gt is None) and confidences is None:
        fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.5)
        legend_anchor = (0.5, -0.5)
        bar_ax = fig.add_subplot(gs[:, :])
    elif (pred is None or gt is None) and confidences is not None:
        bar_ax = fig.add_subplot(gs[0:3])
        legend_anchor = (0.5, -0.5)
    elif (pred is not None and gt is not None) and confidences is None:
        bar_ax = fig.add_subplot(gs[0:3])
        legend_anchor = (0.5, -0.5)
    elif (pred is not None and gt is not None) and confidences is not None:
        fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.2)
        bar_ax = fig.add_subplot(gs[0:2])
        legend_anchor = (0.5, -0.7)

    if palette is None:
        palette = template(num_classes, "cividis")
    if dynamic_palette:
        _palette = template(len(unique_label), "cividis")
        for i, label in enumerate(unique_label):
            palette[label] = _palette[i]

    target_bar = []
    for i in range(max_num_segments):
        if pred is not None and gt is None:
            p = bar_ax.barh(
                "Pred",
                pred_segments[i][1][1] - pred_segments[i][1][0],
                color=palette[pred_segments[i][0]],
                left=acc[0],
            )
            acc[0] += pred_segments[i][1][1] - pred_segments[i][1][0]
            if pred_segments[i][0] not in [x[0] for x in target_bar]:
                target_bar.append((pred_segments[i][0], p[0]))
        elif pred is None and gt is not None:
            p = bar_ax.barh(
                "GT",
                gt_segments[i][1][1] - gt_segments[i][1][0],
                color=palette[gt_segments[i][0]],
                left=acc[0],
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
    ordered_target_bar = sum(
        [target_bar[i::legend_ncols] for i in range(legend_ncols)], []
    )
    ordered_unique_label = np.concatenate(
        [unique_label[i::legend_ncols] for i in range(legend_ncols)]
    )
    bar_ax.legend(
        [x[1] for x in ordered_target_bar],
        [
            int_to_text[int(ordered_unique_label[i])]
            for i in range(len(ordered_unique_label))
        ],
        bbox_to_anchor=legend_anchor,
        loc="upper center",
        ncols=legend_ncols,
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
    if is_jupyter:
        from IPython.display import display

        display(fig)
        plt.close(fig)
        return
    if return_canvas:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

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


def make_video(
    pred: ndarray | None = None,
    gt: ndarray | None = None,
    confidences: ndarray | None = None,
    image_dir: str | None = None,
    images: list[str] | list[np.ndarray] | None = None,
    video_path: str | None = None,
    out_path: str = "action_segmentation.mp4",
    backgrounds: ndarray = np.array([]),
    int_to_text: dict[int, str] = dict(),
    num_classes: int = 50,
    show_segment: bool = True,
    show_label: bool = True,
    legend_ncols: int = 7,
):
    reader = VideoReader(image_dir=image_dir, images=images, video_path=video_path)
    video_size = reader.image_size
    text_area_size = (80, video_size[1])

    if show_segment:
        segmentation = plot_action_segmentation(
            pred=pred,
            gt=gt,
            confidences=confidences,
            backgrounds=backgrounds,
            num_classes=num_classes,
            return_canvas=True,
            int_to_text=int_to_text,
            legend_ncols=legend_ncols,
        )
        segmentation = cv2.resize(
            segmentation,
            None,
            fx=video_size[1] / segmentation.shape[1],
            fy=video_size[1] / segmentation.shape[1],
        )
        video_size = (video_size[0] + segmentation.shape[0], video_size[1])
        bar_width = max(5, int(segmentation.shape[1] * 0.005))

    if show_label:
        video_size = (video_size[0] + text_area_size[0], video_size[1])

    with VideoWriter(
        filename=out_path,
        framerate=30,
        size=video_size,
    ) as writer:
        for i, image in enumerate(tqdm(reader, total=reader.num_frames)):
            if show_segment:
                seg = segmentation.copy()
                segment_offset = 90
                bar_left_pos = segment_offset + int(
                    i
                    / reader.num_frames
                    * (seg.shape[1] - segment_offset * 2 - bar_width)
                )
                bar_start = bar_left_pos
                bar_end = bar_left_pos + bar_width
                seg[55 : seg.shape[0] // 2, bar_start:bar_end] = 250
                image = np.concatenate([image, seg], axis=0)
            if show_label:
                seg = np.full(
                    (text_area_size[0], image.shape[1], 3), 255, dtype=np.uint8
                )
                if pred is not None and gt is None:
                    cv2.putText(
                        img=seg,
                        text="Pred: " + int_to_text[pred[i]],
                        org=(
                            text_area_size[1] // 4,
                            text_area_size[0] // 2 + 20,
                        ),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=(0, 0, 0),
                        thickness=5,
                    )
                if pred is None and gt is not None:
                    cv2.putText(
                        img=seg,
                        text="GT: " + int_to_text[gt[i]],
                        org=(
                            text_area_size[1] // 4,
                            text_area_size[0] // 2 + 20,
                        ),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=(0, 0, 0),
                        thickness=5,
                    )
                if pred is not None and gt is not None:
                    cv2.putText(
                        img=seg,
                        text="Pred: " + int_to_text[pred[i]],
                        org=(
                            text_area_size[1] // 4,
                            text_area_size[0] // 2 + 20,
                        ),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=(0, 0, 0),
                        thickness=5,
                    )
                    cv2.putText(
                        img=seg,
                        text="GT: " + int_to_text[gt[i]],
                        org=(
                            text_area_size[1] // 4 * 3,
                            text_area_size[0] // 2 + 20,
                        ),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=(0, 0, 0),
                        thickness=5,
                    )
                image = np.concatenate([image, seg], axis=0)

            writer.update(image)


class Visualizer(Base):
    metrics: dict[str, list[tuple[int, float]]] = dict()

    def __init__(self):
        super().__init__(name="Visualizer")

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
            plot_metrics(key, epochs, metrics, f"{save_dir}/{key}.png")
