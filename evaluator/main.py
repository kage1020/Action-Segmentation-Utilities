import numpy as np
import torch


class Evaluator:
    taus: list[float]
    backgrounds: list[int]
    num_videos: int
    num_total_frames: int
    num_correct_frames: int
    edit_distances: int
    tps: list[int]
    fps: list[int]
    fns: list[int]

    def __init__(self, taus: list[float] = [0.1, 0.25, 0.5], backgrounds: list[int] = []):
        self.taus = taus
        self.backgrounds = backgrounds
        self.num_videos = 0
        self.num_total_frames = 0
        self.num_correct_frames = 0
        self.edit_distances = 0
        self.tps = [0] * len(taus)
        self.fps = [0] * len(taus)
        self.fns = [0] * len(taus)

    @staticmethod
    def to_np(x: list | np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, list):
            return np.array(x)
        raise ValueError("Invalid input type")

    @staticmethod
    def to_segments(x: list | np.ndarray | torch.Tensor, backgrounds: list | np.ndarray | torch.Tensor) -> list:
        _x = Evaluator.to_np(x)
        _backgrounds = Evaluator.to_np(backgrounds)
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
    def accuracy_frame(gt: list | np.ndarray | torch.Tensor, pred: list | np.ndarray | torch.Tensor) -> float:
        _gt = Evaluator.to_np(gt)
        _pred = Evaluator.to_np(pred)
        return np.mean(_gt == _pred)

    @staticmethod
    def accuracy_class(gt: list | np.ndarray | torch.Tensor, pred: list | np.ndarray | torch.Tensor) -> list[float]:
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
    def levenshtein(x: list | np.ndarray | torch.Tensor, y: list | np.ndarray | torch.Tensor) -> int:
        n = len(x)
        m = len(y)
        dp = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] + 1
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j - 1] + 1
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + (x[i - 1] != y[j - 1])
                )
        return dp[n][m]

    @staticmethod
    def edit_score(gt: list | np.ndarray | torch.Tensor, pred: list | np.ndarray | torch.Tensor, backgrounds: list | np.ndarray | torch.Tensor) -> float:
        _gt = Evaluator.to_np(gt)
        _pred = Evaluator.to_np(pred)
        gt_segments = Evaluator.to_segments(_gt, backgrounds)
        pred_segments = Evaluator.to_segments(_pred, backgrounds)
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
    def tp_fp_fn(gt: list | np.ndarray | torch.Tensor, pred: list | np.ndarray | torch.Tensor, tau: float, backgrounds: list | np.ndarray | torch.Tensor) -> tuple[int, int, int]:
        _gt = Evaluator.to_np(gt)
        _pred = Evaluator.to_np(pred)
        gt_segments = Evaluator.to_segments(_gt, backgrounds)
        pred_segments = Evaluator.to_segments(_pred, backgrounds)
        tp = 0
        fp = 0
        fn = 0
        hits = set()
        for pred_segment in pred_segments:
            matches = [gt_segment for gt_segment in gt_segments if pred_segment[0] == gt_segment[0]]
            ious = [Evaluator.iou(gt_segment[1], pred_segment[1]) for gt_segment in matches]
            idx = np.argmax(ious) if len(ious) > 0 else None
            if idx is not None and ious[idx] >= tau and matches[idx] not in hits:
                tp += 1
                hits.add(matches[idx])
            else:
                fp += 1
        fn = len(gt_segments) - len(hits)
        return tp, fp, fn

    @staticmethod
    def precision(gt: list | np.ndarray | torch.Tensor, pred: list | np.ndarray | torch.Tensor, tau: float, backgrounds: list | np.ndarray | torch.Tensor) -> float:
        tp, fp, _ = Evaluator.tp_fp_fn(gt, pred, tau, backgrounds)
        return tp / (tp + fp) if tp + fp > 0 else 0.0

    @staticmethod
    def recall(gt: list | np.ndarray | torch.Tensor, pred: list | np.ndarray | torch.Tensor, tau: float, backgrounds: list | np.ndarray | torch.Tensor) -> float:
        tp, _, fn = Evaluator.tp_fp_fn(gt, pred, tau, backgrounds)
        return tp / (tp + fn) if tp + fn > 0 else 0.0

    @staticmethod
    def f1(gt: list | np.ndarray | torch.Tensor, pred: list | np.ndarray | torch.Tensor, tau: float, backgrounds: list | np.ndarray | torch.Tensor) -> float:
        p = Evaluator.precision(gt, pred, tau, backgrounds)
        r = Evaluator.recall(gt, pred, tau, backgrounds)
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    def add(self, gt: list | np.ndarray | torch.Tensor, pred: list | np.ndarray | torch.Tensor, tau: float, backgrounds: list | np.ndarray | torch.Tensor) -> None:
        self.num_videos += 1
        self.num_total_frames += len(gt)
        self.num_correct_frames += round(len(gt) * Evaluator.accuracy_frame(gt, pred))
        self.edit_distances += Evaluator.edit_score(gt, pred, backgrounds)
        tps, fps, fns = zip(*[Evaluator.tp_fp_fn(gt, pred, tau, backgrounds) for tau in self.taus])
        self.tps = [self.tps[i] + tps[i] for i in range(len(self.tps))]
        self.fps = [self.fps[i] + fps[i] for i in range(len(self.fps))]
        self.fns = [self.fns[i] + fns[i] for i in range(len(self.fns))]

    def get(self):
        acc = self.num_correct_frames / self.num_total_frames * 100
        edit = self.edit_distances / self.num_videos
        precision = [self.tps[i] / (self.tps[i] + self.fps[i]) if self.tps[i] + self.fps[i] > 0 else 0.0 for i in range(len(self.tps))]
        recall = [self.tps[i] / (self.tps[i] + self.fns[i]) if self.tps[i] + self.fns[i] > 0 else 0.0 for i in range(len(self.tps))]
        f1 = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0.0 for i in range(len(precision))]
        return acc, edit, f1

    def reset(self):
        self.num_videos = 0
        self.num_total_frames = 0
        self.num_correct_frames = 0
        self.edit_distances = 0
        self.tps = [0] * len(self.tps)
        self.fps = [0] * len(self.fps)
        self.fns = [0] * len(self.fns)


if __name__ == '__main__':
    print(Evaluator.levenshtein([1, 2, 3, 4, 5], [1, 2, 3, 5]))
