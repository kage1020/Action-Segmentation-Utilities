import glob
import os
from tqdm import tqdm
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from trainer import PaprikaConfig


def get_step_des_feats(cfg: PaprikaConfig, language_model="MPNet"):
    if language_model == "MPNet":
        step_des_feats = np.load(Path(cfg.document_dir) / "mpnet_feat.npy")

    elif language_model == "S3D":
        with open(
            Path(cfg.document_dir) / "s3d_text_feat" / "step_embeddings.pickle", "rb"
        ) as f:
            step_des_feats = pickle.load(f)

    return step_des_feats


def get_all_video_ids(cfg: PaprikaConfig):
    video_ids = Path(cfg.dataset.base_dir) / cfg.dataset.name / "video_IDs.npy"
    if video_ids.exists():
        videos = np.load(video_ids)
    else:
        videos = []
        video_features = glob.glob(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/features/*"
        )
        for f in tqdm(video_features):
            videos.append(Path(f).stem)
        np.save(video_ids, videos)

    return videos


def find_matching_of_a_segment_given_sorted_val_corres_idx(
    sorted_values, sorted_indices, criteria="threshold", threshold=0.7, topK=3
):
    matched_steps = list()
    matched_steps_score = list()
    if criteria == "threshold":
        # Pick all steps with sim-score > threshold.
        for i in range(len(sorted_values)):
            if sorted_values[i] > threshold:
                matched_steps.append(sorted_indices[i])
                matched_steps_score.append(sorted_values[i])

    elif criteria == "threshold+topK":
        # From the ones with sim-score > threshold,
        # pick the top K if existing.
        for i in range(len(sorted_values)):
            if sorted_values[i] > threshold:
                if len(matched_steps) < topK:
                    matched_steps.append(sorted_indices[i])
                    matched_steps_score.append(sorted_values[i])
                else:
                    break

    elif criteria == "topK":
        # Pick the top K
        for i in range(len(sorted_indices)):
            if len(matched_steps) < topK:
                matched_steps.append(sorted_indices[i])
                matched_steps_score.append(sorted_values[i])
            else:
                break

    else:
        print(
            "The criteria is not implemented!\nFunc: {}\nFile:{}".format(
                __name__, __file__
            )
        )
        os._exit(0)

    return matched_steps, matched_steps_score


def find_matching_of_a_segment(sim_scores, criteria="threshold", threshold=0.7, topK=3):
    sorted_values = np.sort(sim_scores)[::-1]  # sort in descending order
    sorted_indices = np.argsort(-sim_scores)  # indices of sorting in descending order

    matched_steps, matched_steps_score = (
        find_matching_of_a_segment_given_sorted_val_corres_idx(
            sorted_values,
            sorted_indices,
            criteria=criteria,
            threshold=threshold,
            topK=topK,
        )
    )

    return matched_steps, matched_steps_score


def cos_sim(a, b):
    return torch.mm(F.normalize(a, p=2, dim=1), F.normalize(b, p=2, dim=1).transpose(0, 1))
