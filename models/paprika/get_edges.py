from pathlib import Path
import json
import glob
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
from trainer import PaprikaConfig
from .helper import get_all_video_ids, find_matching_of_a_segment
from .get_nodes import get_nodes


def get_edges_between_document_steps_in_document(cfg: PaprikaConfig):
    with open(Path(cfg.document_dir) / "step_label_text.json", "r") as f:
        document = json.load(f)

    step_id = 0
    article_po_to_step_id = dict()
    for article_id in range(len(document)):
        for article_step_idx in range(len(document[article_id])):
            article_po_to_step_id[(article_id, article_step_idx)] = step_id
            step_id += 1
    total_num_steps = len(article_po_to_step_id)

    document_steps_1hop_edges = np.zeros((total_num_steps, total_num_steps))
    for article_id in range(len(document)):
        for article_step_idx in range(1, len(document[article_id])):
            predecessor = article_po_to_step_id[(article_id, article_step_idx - 1)]
            successor = article_po_to_step_id[(article_id, article_step_idx)]

            document_steps_1hop_edges[predecessor, successor] += 1

    return document_steps_1hop_edges


def get_edges_between_document_steps_of_one_dataset_video(
    cfg: PaprikaConfig, video, sim_score_path
):
    sim_score_paths_of_segments_this_video = sorted(
        glob.glob(Path(sim_score_path) / video / "segment_*.npy")
    )

    edges_meta = list()
    # loop over segments
    for video_segment_idx in range(1, len(sim_score_paths_of_segments_this_video)):
        segment_pre_sim_scores = np.load(
            sim_score_paths_of_segments_this_video[video_segment_idx - 1]
        )
        segment_suc_sim_scores = np.load(
            sim_score_paths_of_segments_this_video[video_segment_idx]
        )

        predecessors, _ = find_matching_of_a_segment(
            segment_pre_sim_scores,
            criteria=cfg.graph_find_matched_steps_criteria,
            threshold=cfg.graph_find_matched_steps_for_segments_thresh,
            topK=cfg.graph_find_matched_steps_for_segments_topK,
        )

        successors, _ = find_matching_of_a_segment(
            segment_suc_sim_scores,
            criteria=cfg.graph_find_matched_steps_criteria,
            threshold=cfg.graph_find_matched_steps_for_segments_thresh,
            topK=cfg.graph_find_matched_steps_for_segments_topK,
        )

        for predecessor in predecessors:
            for successor in successors:
                if predecessor != successor:  # a step transition
                    edges_meta.append(
                        [
                            predecessor,
                            successor,
                            segment_pre_sim_scores[predecessor]
                            * segment_suc_sim_scores[successor],
                        ]
                    )
    return edges_meta


def threshold_and_normalize(G, edge_min_aggconf=1000):
    G_new = np.zeros((G.shape[0], G.shape[0]))
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i, j] > edge_min_aggconf:
                G_new[i, j] = G[i, j]
    G = G_new

    G_flat = G.reshape(
        G.shape[0] * G.shape[0],
    )
    x = [np.log(val) for val in G_flat if val != 0]
    assert (
        len(x) > 0
    ), "No edges remain after thresholding! Please use a smaller edge_min_aggconf!"
    max_val = np.max(x)

    G_new = np.zeros((G.shape[0], G.shape[0]))
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i, j] > 0:
                G_new[i, j] = (np.log(G[i, j]) - 0) / (max_val - 0)  # log min max norm
    G = G_new
    return G


def get_edges_between_document_steps_in_dataset(cfg: PaprikaConfig, total_num_steps):
    sim_score_path = Path(cfg.dataset.base_dir) / cfg.dataset.name / "sim_scores"

    videos = get_all_video_ids(cfg)

    edges_metas = list()
    for video in videos:
        edges_metas.append(
            get_edges_between_document_steps_of_one_dataset_video(
                cfg, video, sim_score_path
            )
        )

    dataset_steps_1hop_edges = np.zeros((total_num_steps, total_num_steps))
    for edges_meta in edges_metas:
        for [predecessor, successor, confidence] in edges_meta:
            dataset_steps_1hop_edges[predecessor, successor] += confidence

    dataset_steps_1hop_edges = threshold_and_normalize(
        dataset_steps_1hop_edges, cfg.edge_min_aggconf
    )

    return dataset_steps_1hop_edges


def get_node_transition_candidates(step2node, G_wikihow, G_howto100m):
    candidates = dict(list)

    for step_id in tqdm(range(len(step2node))):
        for direct_outstep_id in G_wikihow[step_id].indices:
            conf = G_wikihow[step_id, direct_outstep_id]
            node_id = step2node[step_id]
            direct_outnode_id = step2node[direct_outstep_id]
            candidates[(node_id, direct_outnode_id)].append(conf)

    for step_id in tqdm(range(len(step2node))):
        for direct_outstep_id in G_howto100m[step_id].indices:
            conf = G_howto100m[step_id, direct_outstep_id]
            node_id = step2node[step_id]
            direct_outnode_id = step2node[direct_outstep_id]
            candidates[(node_id, direct_outnode_id)].append(conf)

    return candidates


def keep_highest_conf_for_each_candidate(candidates):
    edges = dict()
    for node_id, direct_outnode_id in tqdm(candidates):
        max_conf = np.max(candidates[(node_id, direct_outnode_id)])

        edges[(node_id, direct_outnode_id)] = max_conf
    return edges


def build_pkg_adj_matrix(edges, num_nodes):
    pkg = np.zeros((num_nodes, num_nodes))
    for node_id, direct_outnode_id in tqdm(edges):
        pkg[node_id, direct_outnode_id] = edges[(node_id, direct_outnode_id)]
    return pkg


def get_edges(cfg: PaprikaConfig):
    # --  get the edges between step headlines
    G_document = get_edges_between_document_steps_in_document(cfg)
    G_dataset = get_edges_between_document_steps_in_dataset(cfg, G_document.shape[0])
    G_document_csr, G_howto100m_csr = csr_matrix(G_document), csr_matrix(G_dataset)

    # -- turn edges between step headlines into edges between nodes
    node2step, step2node = get_nodes(cfg)
    node_transition_candidates = get_node_transition_candidates(
        step2node, G_document_csr, G_howto100m_csr
    )
    pkg_edges = keep_highest_conf_for_each_candidate(node_transition_candidates)
    pkg = build_pkg_adj_matrix(pkg_edges, len(node2step))

    return pkg, G_document, G_dataset
