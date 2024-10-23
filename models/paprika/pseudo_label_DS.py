import os
import numpy as np
from tqdm import tqdm
import pickle

from trainer import PaprikaConfig
from .helper import find_matching_of_a_segment
from .get_samples import get_samples


def get_pseudo_label_DS_for_one_segment(cfg: PaprikaConfig, sample_gt_path):
    step_scores = np.load(sample_gt_path)
    matched_steps, matched_steps_scores = find_matching_of_a_segment(
        step_scores,
        criteria=cfg.label_find_matched_steps_criteria,
        threshold=cfg.label_find_matched_steps_for_segments_thresh,
        topK=cfg.label_find_matched_steps_for_segments_topK,
    )

    pseudo_label_DS = {"indices": matched_steps, "values": matched_steps_scores}
    return pseudo_label_DS


def get_pseudo_label_DS(cfg: PaprikaConfig):
    if not os.path.exists(os.path.join(cfg.howto100m_dir, "samples/samples.pickle")):
        get_samples(cfg)
    with open(os.path.join(cfg.howto100m_dir, "samples/samples.pickle"), "rb") as f:
        samples = pickle.load(f)

    sim_score_path = os.path.join(cfg.howto100m_dir, "DS_sim_scores")
    sample_pseudo_label_savedir = os.path.join(cfg.howto100m_dir, "DS_pseudo_labels")
    os.makedirs(sample_pseudo_label_savedir, exist_ok=True)

    sample_pseudo_label_savepath = os.path.join(
        sample_pseudo_label_savedir,
        "DS-criteria_{}-threshold_{}-topK_{}-size_{}.pickle".format(
            cfg.label_find_matched_steps_criteria,
            cfg.label_find_matched_steps_for_segments_thresh,
            cfg.label_find_matched_steps_for_segments_topK,
            cfg.num_steps,
        ),
    )

    if not os.path.exists(sample_pseudo_label_savepath):
        # start processing
        pseudo_label_DS = dict()
        for sample_index in tqdm(range(len(samples))):
            (video_sid, segment_iid) = samples[sample_index]
            segment_sid = "segment_{}".format(segment_iid)
            sample_gt_path = os.path.join(
                sim_score_path, video_sid, segment_sid + ".npy"
            )

            pseudo_label_DS[sample_index] = get_pseudo_label_DS_for_one_segment(
                cfg, sample_gt_path
            )

        # save
        with open(sample_pseudo_label_savepath, "wb") as f:
            pickle.dump(pseudo_label_DS, f)

    return
