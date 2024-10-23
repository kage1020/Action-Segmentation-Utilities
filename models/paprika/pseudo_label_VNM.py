import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
from trainer import PaprikaConfig
from .get_nodes import get_nodes
from .get_samples import get_samples
from .helper import find_matching_of_a_segment


def get_pseudo_label_VNM_for_one_segment(
    cfg: PaprikaConfig, node2step, step2node, sample_gt_path
):
    step_scores = np.load(sample_gt_path)
    # obtain node scores
    node_scores = dict()
    for node_id in range(len(node2step)):
        node_scores[node_id] = 0
    for step_id in range(len(step_scores)):
        node_id = step2node[step_id]
        node_scores[node_id] = max(node_scores[node_id], step_scores[step_id])

    node_scores_arr = np.array([node_scores[node_id] for node_id in node_scores])

    matched_nodes, matched_nodes_scores = find_matching_of_a_segment(
        node_scores_arr,
        criteria=cfg.label_find_matched_nodes_criteria,
        threshold=cfg.label_find_matched_nodes_for_segments_thresh,
        topK=cfg.label_find_matched_nodes_for_segments_topK,
    )

    pseudo_label_VNM = {"indices": matched_nodes, "values": matched_nodes_scores}
    return pseudo_label_VNM


def get_pseudo_label_VNM(cfg: PaprikaConfig):
    sample_path = (
        Path(cfg.dataset.base_dir) / cfg.dataset.name / "samples" / "samples.pickle"
    )
    if not sample_path.exists():
        get_samples(cfg)
    with open(sample_path, "rb") as f:
        samples = pickle.load(f)

    node2step, step2node = get_nodes(cfg)

    sim_score_path = Path(cfg.dataset.base_dir) / cfg.dataset.name / "sim_scores"
    sample_pseudo_label_savedir = (
        Path(cfg.dataset.base_dir) / cfg.dataset.name / "pseudo_labels"
    )
    sample_pseudo_label_savedir.mkdir(parents=True, exist_ok=True)

    sample_pseudo_label_savepath = (
        Path(sample_pseudo_label_savedir)
        / f"VNM-criteria_{cfg.label_find_matched_nodes_criteria}-threshold_{cfg.label_find_matched_nodes_for_segments_thresh}-topK_{cfg.label_find_matched_nodes_for_segments_topK}-size_{cfg.num_nodes}.pickle"
    )

    if not sample_pseudo_label_savepath.exists():
        # start processing
        pseudo_label_VNM = dict()
        for sample_index in tqdm(range(len(samples))):
            (video_sid, segment_iid) = samples[sample_index]
            segment_sid = f"segment_{segment_iid}"
            sample_gt_path = Path(sim_score_path) / video_sid / f"{segment_sid}.npy"

            pseudo_label_VNM[sample_index] = get_pseudo_label_VNM_for_one_segment(
                cfg, node2step, step2node, sample_gt_path
            )

        # save
        with open(sample_pseudo_label_savepath, "wb") as f:
            pickle.dump(pseudo_label_VNM, f)
