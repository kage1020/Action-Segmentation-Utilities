from typing import Literal
import numpy as np
from sklearn.metrics import roc_auc_score
from asu.base.main import Base, Config

from numpy import ndarray
from torch import Tensor

# TODO: modify


class Evaluator(Base):
    def __init__(
        self,
        cfg: Config,
        taus: tuple[float, float, float] = (0.1, 0.25, 0.5),
    ):
        super().__init__(
            name="Evaluator",
            mapping_path=f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.mapping_path}",
            actions_path=f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.actions_path}",
            matching_path=f"{cfg.dataset.base_dir}/{cfg.dataset.name}/{cfg.dataset.matching_path}",
            has_mapping_header=cfg.dataset.has_mapping_header,
            has_actions_header=cfg.dataset.has_actions_header,
            mapping_separator=cfg.dataset.mapping_separator,
            actions_class_separator=cfg.dataset.actions_class_separator,
            actions_action_separator=cfg.dataset.actions_action_separator,
            matching_separator=cfg.dataset.matching_separator,
            backgrounds=cfg.dataset.backgrounds,
        )
        self.taus = taus
        self.num_videos = 0
        self.num_total_frames = 0
        self.num_correct_frames = 0
        self.edit_distances = 0
        self.tps = [0] * len(taus)
        self.fps = [0] * len(taus)
        self.fns = [0] * len(taus)
        self.aucs = []

    @staticmethod
    def accuracy_frame(
        gt: list | ndarray | Tensor, pred: list | ndarray | Tensor
    ) -> float:
        _gt = Evaluator.to_np(gt)
        _pred = Evaluator.to_np(pred)
        return np.mean(_gt == _pred)

    @staticmethod
    def accuracy_class(
        gt: list | ndarray | Tensor, pred: list | ndarray | Tensor
    ) -> list[float]:
        _gt = Evaluator.to_np(gt)
        _pred = Evaluator.to_np(pred)
        classes = np.unique(np.concatenate([_gt, _pred]))
        classes.sort()
        acc = []
        for c in classes:
            mask = _gt == c
            acc.append(np.mean(_gt[mask] == _pred[mask]) if np.any(mask) else 0.0)
        return acc

    @staticmethod
    def levenshtein(x: list | ndarray | Tensor, y: list | ndarray | Tensor) -> int:
        m = len(x)
        n = len(y)
        dp = np.zeros((m + 1, n + 1))
        for i in range(1, m + 1):
            dp[i, 0] = i
        for i in range(1, n + 1):
            dp[0, i] = i
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[j, i] = min(
                    dp[j - 1, i] + 1,
                    dp[j, i - 1] + 1,
                    dp[j - 1, i - 1] + int(x[j - 1] != y[i - 1]),
                )
        return dp[m][n]

    @staticmethod
    def edit_score(
        gt: list | ndarray | Tensor,
        pred: list | ndarray | Tensor,
        backgrounds: list | ndarray | Tensor = [],
    ) -> float:
        _gt = Evaluator.to_np(gt)
        _pred = Evaluator.to_np(pred)
        _backgrounds = Evaluator.to_np(backgrounds)
        gt_segments = Evaluator.to_segments(_gt, _backgrounds)
        pred_segments = Evaluator.to_segments(_pred, _backgrounds)
        if len(gt_segments) == 0 or len(pred_segments) == 0:
            return 0.0
        x, _ = zip(*gt_segments)
        y, _ = zip(*pred_segments)
        return (1 - Evaluator.levenshtein(x, y) / max(len(x), len(y))) * 100

    @staticmethod
    def iou(gt: tuple[int, int], pred: tuple[int, int]) -> float:
        intersection = max(0, min(gt[1], pred[1]) - max(gt[0], pred[0]))
        union = max(gt[1], pred[1]) - min(gt[0], pred[0])
        return intersection / union

    @staticmethod
    def tp_fp_fn(
        gt: list | ndarray | Tensor,
        pred: list | ndarray | Tensor,
        tau: float,
        backgrounds: list | ndarray | Tensor = [],
    ) -> tuple[int, int, int]:
        _gt = Evaluator.to_np(gt)
        _pred = Evaluator.to_np(pred)
        _backgrounds = Evaluator.to_np(backgrounds)
        gt_segments = Evaluator.to_segments(_gt, _backgrounds)
        pred_segments = Evaluator.to_segments(_pred, _backgrounds)
        tp = 0
        fp = 0
        fn = 0
        hits = set()
        for pred_segment in pred_segments:
            matches = [
                gt_segment
                for gt_segment in gt_segments
                if pred_segment[0] == gt_segment[0]
            ]
            ious = [
                Evaluator.iou(gt_segment[1], pred_segment[1]) for gt_segment in matches
            ]
            idx = np.argmax(ious) if len(ious) > 0 else None
            if idx is not None and ious[idx] >= tau and matches[idx] not in hits:
                tp += 1
                hits.add(matches[idx])
            else:
                fp += 1
        fn = len(gt_segments) - len(hits)
        return tp, fp, fn

    @staticmethod
    def precision(
        gt: list | ndarray | Tensor,
        pred: list | ndarray | Tensor,
        tau: float,
    ) -> float:
        tp, fp, _ = Evaluator.tp_fp_fn(gt, pred, tau)
        return tp / (tp + fp) if tp + fp > 0 else 0.0

    @staticmethod
    def recall(
        gt: list | ndarray | Tensor,
        pred: list | ndarray | Tensor,
        tau: float,
    ) -> float:
        tp, _, fn = Evaluator.tp_fp_fn(gt, pred, tau)
        return tp / (tp + fn) if tp + fn > 0 else 0.0

    @staticmethod
    def f1(
        gt: list | ndarray | Tensor,
        pred: list | ndarray | Tensor,
        tau: float,
    ) -> float:
        p = Evaluator.precision(gt, pred, tau)
        r = Evaluator.recall(gt, pred, tau)
        return 200 * p * r / (p + r) if p + r > 0 else 0.0

    @staticmethod
    def auc(
        gt: list | ndarray | Tensor,
        prob: ndarray | Tensor,
        backgrounds: list | ndarray | Tensor,
    ):
        _gt = Evaluator.to_np(gt)
        _prob = Evaluator.to_np(prob)
        _backgrounds = Evaluator.to_np(backgrounds)
        mask = np.isin(_gt, _backgrounds)
        _gt[mask] = -100
        _prob[mask] = 0
        _prob[mask, 0] = 1
        return roc_auc_score(_gt, _prob, multi_class="ovo")

    def add(
        self,
        gt: ndarray | Tensor,
        pred: ndarray | Tensor,
        prob: ndarray | Tensor | None = None,
    ) -> None:
        self.num_videos += 1
        self.num_total_frames += len(gt)
        self.num_correct_frames += (gt == pred).sum()
        self.edit_distances += Evaluator.edit_score(gt, pred, self.backgrounds)
        tps, fps, fns = zip(
            *[Evaluator.tp_fp_fn(gt, pred, tau, self.backgrounds) for tau in self.taus]
        )
        self.tps = [self.tps[i] + tps[i] for i in range(len(self.tps))]
        self.fps = [self.fps[i] + fps[i] for i in range(len(self.fps))]
        self.fns = [self.fns[i] + fns[i] for i in range(len(self.fns))]
        if prob:
            self.aucs.append(Evaluator.auc(gt, prob, self.backgrounds))

    def get(
        self,
        task: Literal[
            "action_segmentation", "anomaly_detection"
        ] = "action_segmentation",
    ) -> tuple[float, float, list[float]] | float:
        if task == "action_segmentation":
            acc = self.num_correct_frames / self.num_total_frames * 100
            edit = self.edit_distances / self.num_videos
            precision = [
                self.tps[i] / (self.tps[i] + self.fps[i])
                if self.tps[i] + self.fps[i] > 0
                else 0.0
                for i in range(len(self.tps))
            ]
            recall = [
                self.tps[i] / (self.tps[i] + self.fns[i])
                if self.tps[i] + self.fns[i] > 0
                else 0.0
                for i in range(len(self.tps))
            ]
            f1 = [
                200 * precision[i] * recall[i] / (precision[i] + recall[i])
                if precision[i] + recall[i] > 0
                else 0.0
                for i in range(len(precision))
            ]
            return acc, edit, f1
        elif task == "anomaly_detection":
            return np.mean(self.aucs)

    def reset(self) -> None:
        self.num_videos = 0
        self.num_total_frames = 0
        self.num_correct_frames = 0
        self.edit_distances = 0
        self.tps = [0] * len(self.tps)
        self.fps = [0] * len(self.fps)
        self.fns = [0] * len(self.fns)
        self.aucs = []

    def save_labels(self, x: list, path: str) -> None:
        with open(path, "w") as f:
            f.writelines([f"{item}\n" for item in x])

    def save_np(self, x: ndarray, path: str) -> None:
        np.save(path, x)
