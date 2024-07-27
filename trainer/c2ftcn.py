import os
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from base import Config, Base
from .main import Trainer
from loader import C2FTCNBreakfastDataLoader

# TODO: modify


@dataclass
class C2FTCNConfig(Config):
    feature_size: int
    output_size: int
    num_iter: int
    start_iter: int
    unsupervised_skip: bool
    epochs_unsupervised: int
    semi_per: float
    lr_proj: float
    lr_main: float
    lr_unsupervised: float
    gamma_proj: float
    gamma_main: float
    steps_proj: list[int]
    steps_main: list[int]
    epsilon: float
    epsilon_l: float
    delta: float
    weights: list[float]
    high_level_act_loss: bool
    num_samples_frames: int


class C2FTCNCriterion(Module):
    def __init__(self, cfg: C2FTCNConfig | Config, num_classes: int, mse_weight: float):
        self.cfg = C2FTCNConfig(*cfg)
        self.ce = CrossEntropyLoss(ignore_index=-100)
        self.mse = MSELoss(reduction="none")
        self.num_classes = num_classes
        self.mse_weight = mse_weight

    def __get_mse_loss(self, count: Tensor, pred: Tensor):
        mask = torch.arange(pred.shape[2], device=pred.device)[None, :] < count[:, None]
        mask = mask.to(torch.float32).to(pred.device).unsqueeze(1)
        prev = F.log_softmax(pred[:, :, 1:], dim=1)
        next_ = F.log_softmax(pred.detach()[:, :, :-1], dim=1)
        mse = self.mse(prev, next_)
        loss = self.mse_weight * torch.mean(
            torch.clamp(mse, min=0, max=16) * mask[:, :, 1:]
        )
        return loss

    def __get_unsupervised_loss(
        self, count: Tensor, pred: Tensor, activity_labels, labels
    ):
        vid_ids = []
        f1 = []
        t1 = []
        label_info = []
        feature_activity = []
        max_pool_features = []
        batch_size = pred.shape[0]

        for j in range(batch_size):
            num_frames = count[j]

            split_frames = torch.linspace(
                0, num_frames, self.cfg.num_samples_frames, dtype=torch.int
            )
            idx = []
            for k in range(len(split_frames) - 1):
                start = split_frames[k]
                end = split_frames[k + 1]
                list_frames = list(range(start, end + 1))
                idx.append(np.random.choice(list_frames, 1)[0])

            idx = torch.tensor(idx).type(torch.long).to(pred.device)
            idx = torch.clamp(idx, 0, int(num_frames) - 1)

            v_low = int(np.ceil(self.cfg.epsilon_l * num_frames.item()))
            v_high = int(np.ceil(self.cfg.epsilon * num_frames.item()))
            if v_high <= v_low:
                v_high = v_low + 2
            offset = (
                torch.randint(low=v_low, high=v_high, size=(len(idx),))
                .type(torch.long)
                .to(pred.device)
            )
            prev_idx = torch.clamp(idx - offset, 0, int(num_frames) - 1)

            f1.append(pred[j].permute(1, 0)[idx, :])
            f1.append(pred[j].permute(1, 0)[prev_idx, :])

            if activity_labels is not None:
                feature_activity.extend([activity_labels[j]] * len(idx) * 2)  # type: ignore
            else:
                feature_activity = None

            label_info.append(labels[j][idx])
            label_info.append(labels[j][prev_idx])

            vid_ids.extend([j] * len(idx))
            vid_ids.extend([j] * len(prev_idx))

            idx = idx / num_frames.to(dtype=torch.float32, device=num_frames.device)
            prev_idx = prev_idx / num_frames.to(
                dtype=torch.float32, device=num_frames.device
            )

            t1.extend(idx.detach().cpu().numpy().tolist())
            t1.extend(prev_idx.detach().cpu().numpy().tolist())

            max_pool_features.append(torch.max(pred[j, :, :num_frames], dim=-1)[0])

        vid_ids = torch.tensor(vid_ids).numpy()
        t1 = np.array(t1)
        f1 = torch.cat(f1, dim=0)
        label_info = torch.cat(label_info, dim=0).numpy()

        if feature_activity is not None:
            feature_activity = np.array(feature_activity)

        sim_f1 = f1 @ f1.data.T
        f11 = torch.exp(sim_f1 / 0.1)

        if feature_activity is None:
            pos_weight_mat = torch.tensor(
                (vid_ids[:, None] == vid_ids[None, :])
                & (np.abs(t1[:, None] - t1[None, :]) <= self.cfg.delta)
                & (label_info[:, None] == label_info[None, :])
            )
            negative_samples_minus = (
                torch.tensor(
                    (vid_ids[:, None] == vid_ids[None, :])
                    & (np.abs(t1[:, None] - t1[None, :]) > self.cfg.delta)
                    & (label_info[:, None] == label_info[None, :])
                )
                .type(torch.float32)
                .to(pred.device)
            )
            pos_weight_mat = pos_weight_mat | torch.tensor(
                (vid_ids[:, None] != vid_ids[None, :])
                & (label_info[:, None] == label_info[None, :])
            )
        else:
            pos_weight_mat = torch.tensor(
                (feature_activity[:, None] == feature_activity[None, :])
                & (np.abs(t1[:, None] - t1[None, :]) <= self.cfg.delta)
                & (label_info[:, None] == label_info[None, :])
            )

            negative_samples_minus = (
                torch.tensor(
                    (feature_activity[:, None] == feature_activity[None, :])
                    & (np.abs(t1[:, None] - t1[None, :]) > self.cfg.delta)
                    & (label_info[:, None] == label_info[None, :])
                )
                .type(torch.float32)
                .to(pred.device)
            )

        I = torch.eye(len(pos_weight_mat)).to(pred.device)  # noqa: E741
        pos_weight_mat = (pos_weight_mat).type(torch.float32).to(pred.device) - I
        not_same_activity = 1 - pos_weight_mat - I - negative_samples_minus
        count_pos = torch.sum(pos_weight_mat)

        if count_pos == 0:
            print("Feature level contrast no positive is found")
            feature_contrast_loss = 0
        else:
            feat_sim_pos = pos_weight_mat * f11
            max_val = torch.max(not_same_activity * f11, dim=1, keep_dim=True)[0]  # type: ignore
            acc = torch.sum(feat_sim_pos > max_val) / count_pos
            feat_sim_neg_sum = torch.sum(not_same_activity * f11, dim=1)

            sim_prob = (
                (feat_sim_pos / (feat_sim_neg_sum + feat_sim_pos))
                + not_same_activity
                + I
                + negative_samples_minus
            )

            feature_contrast_loss = -torch.sum(torch.log(sim_prob)) / count_pos

        if self.cfg.high_level_act_loss is True:
            max_pool_features = torch.stack(max_pool_features)
            max_pool_features = max_pool_features / torch.norm(
                max_pool_features, dim=1, keepdim=True
            )
            max_pool_feat_sim = torch.exp(max_pool_features @ max_pool_features.T / 0.1)
            same_activity = torch.tensor(
                np.array(activity_labels)[:, None] == np.array(activity_labels)[None, :]
            )
            I = torch.eye(len(same_activity)).to(pred.device)  # noqa: E741
            same_activity = (same_activity).type(torch.float32).to(pred.device) - I
            not_same_activity = 1 - same_activity - I
            count_pos = torch.sum(same_activity)
            if count_pos == 0:
                print("Video level contrast has no same pairs")
                video_level_contrast = 0
            else:
                max_pool_feat_sim_pos = same_activity * max_pool_feat_sim
                max_pool_feat_sim_neg_sum = torch.sum(
                    not_same_activity * max_pool_feat_sim, dim=1
                )
                sim_prob = (
                    max_pool_feat_sim_pos
                    / (max_pool_feat_sim_neg_sum + max_pool_feat_sim_pos)
                    + not_same_activity
                )
                video_level_contrast = torch.sum(-torch.log(sim_prob + I)) / count_pos
        else:
            video_level_contrast = 0

        unsupervised_loss = feature_contrast_loss + video_level_contrast
        unsupervised_dict_loss = {
            "contrastive_loss": unsupervised_loss,
            "feature_contrast_loss": feature_contrast_loss,
            "video_level_contrast": video_level_contrast,
            "contrast_feature_acc": acc,
        }

        return unsupervised_dict_loss

    def forward(
        self,
        count: Tensor,
        projection: Tensor,
        pred: Tensor,
        labels: Tensor,
        activity_labels: Tensor,
        unsupervised: bool,
    ):
        loss = 0
        if unsupervised is False:
            loss += self.ce(pred, labels)
            loss += self.__get_mse_loss(count, pred)
        else:
            loss = 0

        loss += self.__get_unsupervised_loss(
            count, projection, activity_labels, labels.detach().cpu()
        )["contrastive_loss"]

        return loss


class C2FTCNOptimizer:
    def __init__(self, model: Module, cfg: C2FTCNConfig):
        self.cfg = cfg
        proj_params = (
            set(model.outc0.parameters())
            | set(model.outc1.parameters())
            | set(model.outc2.parameters())
            | set(model.outc3.parameters())
            | set(model.outc4.parameters())
            | set(model.outc5.parameters())
        )
        main_params = set(model.parameters()) - proj_params
        self.optimizers = [
            Adam(list(proj_params), lr=cfg.lr_proj, weight_decay=cfg.weight_decay),
            Adam(list(main_params), lr=cfg.lr_main, weight_decay=cfg.weight_decay),
            Adam(
                list(main_params), lr=cfg.lr_unsupervised, weight_decay=cfg.weight_decay
            ),
        ]

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


class C2FTCNScheduler:
    def __init__(self, optimizers: C2FTCNOptimizer, cfg: C2FTCNConfig):
        self.cfg = cfg
        self.schedulers = [
            MultiStepLR(
                optimizers.optimizers[0],
                milestones=cfg.steps_proj,
                gamma=cfg.gamma_proj,
            ),
            MultiStepLR(
                optimizers.optimizers[1],
                milestones=cfg.steps_main,
                gamma=cfg.gamma_main,
            ),
        ]

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()


class C2FTCNTrainer(Trainer):
    def __init__(
        self,
        cfg: C2FTCNConfig,
        model: Module,
        criterion: Module,
        optimizers: C2FTCNOptimizer,
        schedulers: C2FTCNScheduler,
        evaluator,
    ):
        super().__init__(
            cfg,
            model,
        )
        self.cfg = cfg
        self.criterion = criterion
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.best_acc = 0
        self.best_edit = 0
        self.best_f1 = [0, 0, 0]
        self.evaluator = evaluator

    def dump_gt_labels(self):
        os.makedirs(
            f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.result_dir}/gt",
            exist_ok=True,
        )
        for video_path, gt, _, _ in self.train_loader.dataset.videos:  # type: ignore
            if len(gt) == 0:
                continue

            os.system(
                f"cp {video_path} {self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.result_dir}/{self.cfg.pseudo_dir}/{video_path.name}"
            )

    def dump_labels(self, loader):
        self.model.to(self.device)
        self.model.eval()

        results = {}
        with torch.no_grad():
            for samples, count, _, names in loader:
                samples = samples.to(self.device)

                _, out_pred = self.model(samples, self.cfg.weights)
                out_prob = F.softmax(out_pred, dim=1)
                out_pred = (
                    torch.argmax(out_prob, dim=1).squeeze().detach().cpu().numpy()
                )

                for n, c in zip(names, count):
                    pred = (
                        torch.argmax(out_prob[:, :c], dim=0)
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    if n in results:
                        prev_pred, prev_count = results[n]
                        pred = np.concatenate([prev_pred, pred])
                        c = prev_count + c

                    results[n] = [pred, c]

            os.makedirs(
                f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.result_dir}",
                exist_ok=True,
            )
            for n, (pred, c) in results.items():
                with open(
                    f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.gt_dir}/{n}.txt",
                    "r",
                ) as f:
                    lines = f.readlines()
                    gt = [line.strip() for line in lines if line.strip() != ""]

                pred = [loader.int_to_class[p] for p in pred]
                labels = []

                for i, label in enumerate(pred):
                    st = i * self.cfg.chunk_size
                    end = min(len(gt), st + self.cfg.chunk_size)
                    labels.extend([label] * (end - st))
                    if len(labels) >= len(gt):
                        break

                with open(
                    f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.result_dir}/{self.cfg.pseudo_dir}/{n}.txt",
                    "w",
                ) as f:
                    f.writelines([f"{x}\n" for x in labels])

    def train_supervised(self, train_loader, test_loader):
        self.model.to(self.device)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_loss = 0

            for samples, count, labels, names in train_loader:
                samples = samples.to(self.device)
                count = count.to(self.device)
                labels = labels.to(self.device)

                activity_labels = None
                if self.cfg.dataset == "breakfast":
                    activity_labels = np.array([name.split("_") for name in names])

                projection, pred = self.model(samples, self.cfg.weights)

                loss = self.criterion(
                    count, projection, pred, labels, activity_labels, False
                )
                epoch_loss += loss.item()

                self.optimizers.zero_grad()
                loss.backward()
                self.optimizers.step()
                self.logger.info(f"Epoch {epoch+1} | Loss: {loss.item()}")

                self.train_evaluator.add(labels, pred)

            acc, edit, f1 = self.train_evaluator.get()
            self.logger.info(
                f"Epoch {epoch+1} | F1@10: {f1[0]:.3f}, F1@25: {f1[1]:.3f}, F1@50: {f1[2]:.3f}, Edit: {edit:.3f}, Acc: {acc:.3f}, Loss: {epoch_loss:.3f}"
            )

            if (epoch + 1) % (self.cfg.epochs // 10) == 0:
                Base.save_model(
                    self.model,
                    f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.result_dir}/models/epoch-{epoch+1}.model",
                )

            if self.best_f1[0] < f1[0]:
                self.best_acc = acc
                self.best_edit = edit
                self.best_f1 = f1
                Base.save_model(
                    self.model,
                    f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.result_dir}/models/best.model",
                )

            self.schedulers.step()

            if self.cfg.val_skip:
                continue

            self.model.eval()
            with torch.no_grad():
                for samples, count, labels, names in test_loader:
                    samples = samples.to(self.device)
                    count = count.to(self.device)
                    labels = labels.to(self.device)

                    activity_labels = None
                    if self.cfg.dataset == "breakfast":
                        activity_labels = np.array([name.split("_") for name in names])

                    projection, pred = self.model(samples, self.cfg.weights)
                    self.test_evaluator.add(labels, pred)

            acc, edit, f1 = self.test_evaluator.get()
            self.logger.info(
                f"Epoch {epoch+1} | F1@10: {f1[0]:.3f}, F1@25: {f1[1]:.3f}, F1@50: {f1[2]:.3f}, Edit: {edit:.3f}, Acc: {acc:.3f}, Loss: {epoch_loss:.3f}"
            )

    def train_unsupervised(self, train_loader):
        self.model.to(self.device)

        for epoch in range(self.cfg.epochs_unsupervised):
            self.model.train()
            epoch_loss = 0

            for samples, count, labels, names in train_loader:
                samples = samples.to(self.device)
                count = count.to(self.device)
                labels = labels.to(self.device)

                activity_labels = None
                if self.cfg.dataset == "breakfast":
                    activity_labels = np.array([name.split("_") for name in names])

                projection, pred = self.model(samples, self.cfg.weights)

                loss = self.criterion(
                    count, projection, pred, labels, activity_labels, True
                )
                epoch_loss += loss.item()

                self.optimizers.zero_grad()
                loss.backward()
                self.optimizers.step()
                self.logger.info(f"Epoch {epoch+1} | Loss: {loss.item()}")

            self.schedulers.step()

            if self.cfg.val_skip:
                continue

            self.model.eval()
            with torch.no_grad():
                for samples, count, labels, names in train_loader:
                    samples = samples.to(self.device)
                    count = count.to(self.device)
                    labels = labels.to(self.device)

                    activity_labels = None
                    if self.cfg.dataset == "breakfast":
                        activity_labels = np.array([name.split("_") for name in names])

                    projection, pred = self.model(samples, self.cfg.weights)
                    self.test_evaluator.add(labels, pred)

            acc, edit, f1 = self.test_evaluator.get()
            self.logger.info(
                f"Epoch {epoch+1} | F1@10: {f1[0]:.3f}, F1@25: {f1[1]:.3f}, F1@50: {f1[2]:.3f}, Edit: {edit:.3f}, Acc: {acc:.3f}, Loss: {epoch_loss:.3f}"
            )

    def train(self, train_loader, test_loader):
        for iter_n in range(self.cfg.start_iter, self.cfg.num_iter + 1):
            self.train_supervised(train_loader, test_loader)

            self.model = Base.load_best_model(
                self.model,
                f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.result_dir}/models",
                self.device,
            )

            unlabeled_videos = train_loader.dataset.unlabeled
            unlabeled_loader = DataLoader(
                unlabeled_videos, batch_size=1, shuffle=False, num_workers=4
            )
            self.dump_gt_labels()
            self.dump_labels(unlabeled_loader)

            if iter_n == self.cfg.num_iter:
                break

            unsupervised_train_loader = C2FTCNBreakfastDataLoader(self.cfg, train=True)
            unsupervised_test_loader = C2FTCNBreakfastDataLoader(self.cfg, train=False)
            self.train_unsupervised(unsupervised_train_loader)

        #     if (epoch + 1) % 10 == 0:
        #         torch.save(
        #             self.model.state_dict(),
        #             f"{self.cfg.base_dir}/{self.cfg.result_dir}/epoch-{epoch+1}.model",
        #         )

        #     acc, edit, f1 = self.train_evaluator.get()
        #     self.logger.info(
        #         f"Epoch {epoch+1} | F1@10: {f1[0]:.3f}, F1@25: {f1[1]:.3f}, F1@50: {f1[2]:.3f}, Edit: {edit:.3f}, Acc: {acc:.3f}, Loss: {epoch_loss:.3f}"
        #     )
        #     if self.best_f1[0] < f1[0]:
        #         self.best_acc = acc
        #         self.best_edit = edit
        #         self.best_f1 = f1
        #         torch.save(
        #             self.model.state_dict(),
        #             f"{self.cfg.base_dir}/{self.cfg.result_dir}/best.model",
        #         )
        #     self.train_evaluator.reset()
        #     self.scheduler.step(epoch_loss)

    def test(self, test_loader):
        self.model.to(self.device)
        self.model.eval()
        model_path = f"{self.cfg.base_dir}/{self.cfg.model_dir}/best.model"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        with torch.no_grad():
            for features, mask, labels in test_loader:
                features = features.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features, mask)
                conf, pred = torch.max(F.softmax(outputs[-1], dim=1), dim=1)
                self.test_evaluator.add(labels, pred)
            acc, edit, f1 = self.evaluator.get()
            self.logger.info(
                f"F1@10: {f1[0]:.3f}, F1@25: {f1[1]:.3f}, F1@50: {f1[2]:.3f}, Edit: {edit:.3f}, Acc: {acc:.3f}"
            )
