import pickle
from tqdm import tqdm
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

from trainer import PaprikaConfig
from .get_edges import get_edges
from .helper import find_matching_of_a_segment_given_sorted_val_corres_idx


def get_pseudo_label_NRL_for_one_segment(
    cfg: PaprikaConfig,
    khop,
    in_neighbors_previous_hop,
    out_neighbors_previous_hop,
    pkg,
    pkg_tr,
):
    results = dict()
    for direction_key in ["out", "in"]:  # loop over in & out
        if direction_key == "out":  # out neighbors
            G, neighbors_previous_hop = pkg, out_neighbors_previous_hop
        else:  # in neighors
            G, neighbors_previous_hop = pkg_tr, in_neighbors_previous_hop

        node_scores = dict()
        for i, nei_prehop in enumerate(neighbors_previous_hop["indices"]):
            for direct_nei in G[nei_prehop].indices:
                if khop > 1:
                    node_scores[direct_nei] = (
                        neighbors_previous_hop["values"][i] * G[nei_prehop, direct_nei]
                    )
                else:
                    node_scores[direct_nei] = G[nei_prehop, direct_nei]
        node_scores_sorted = sorted(
            node_scores.items(), key=lambda item: item[1], reverse=True
        )
        # [(node_id, node_score), ... , (node_id, node_score)]

        (matched_neihgbor_nodes, matched_neihgbor_nodes_scores) = (
            find_matching_of_a_segment_given_sorted_val_corres_idx(
                [node_score for (node_id, node_score) in node_scores_sorted],
                [node_id for (node_id, node_score) in node_scores_sorted],
                criteria=cfg.label_find_neighbors_criteria,
                threshold=cfg.label_find_neighbors_thresh,
                topK=cfg.label_find_neighbors_topK,
            )
        )
        results["{}-hop-{}".format(khop, direction_key)] = {
            "indices": matched_neihgbor_nodes,
            "values": matched_neihgbor_nodes_scores,
        }
    return results


def get_pseudo_label_NRL(cfg: PaprikaConfig):
    pkg, _, _ = get_edges(cfg)

    pkg_tr = np.transpose(pkg)
    pkg, pkg_tr = csr_matrix(pkg), csr_matrix(pkg_tr)

    sample_pseudo_label_savedir = (
        Path(cfg.dataset.base_dir) / cfg.dataset.name / "pseudo_labels"
    )
    sample_pseudo_label_savedir.mkdir(parents=True, exist_ok=True)

    for khop in range(1, cfg.label_khop + 1):
        sample_pseudo_label_savepath = (
            Path(sample_pseudo_label_savedir)
            / f"NRL-hop_{khop}-criteria_{cfg.label_find_neighbors_criteria}-threshold_{cfg.label_find_neighbors_thresh}-topK_{cfg.label_find_neighbors_topK}-size_{cfg.num_nodes}.pickle"
        )
        if not sample_pseudo_label_savepath.exists():
            # start processing
            pseudo_label_NRL = dict()
            if khop == 1:
                with open(
                    sample_pseudo_label_savedir
                    / f"VNM-criteria_{cfg.label_find_matched_nodes_criteria}-threshold_{cfg.label_find_matched_nodes_for_segments_thresh}-topK_{cfg.label_find_matched_nodes_for_segments_topK}-size_{cfg.num_nodes}.pickle",
                    "rb",
                ) as f:
                    pseudo_label_VNM = pickle.load(f)

                for sample_index in tqdm(range(len(pseudo_label_VNM))):
                    pseudo_label_NRL[sample_index] = (
                        get_pseudo_label_NRL_for_one_segment(
                            cfg,
                            khop,
                            pseudo_label_VNM[sample_index],
                            pseudo_label_VNM[sample_index],
                            pkg,
                            pkg_tr,
                        )
                    )

            else:
                with open(
                    sample_pseudo_label_savedir
                    / f"NRL-hop_{khop - 1}-criteria_{cfg.label_find_neighbors_criteria}-threshold_{cfg.label_find_neighbors_thresh}-topK_{cfg.label_find_neighbors_topK}-size_{cfg.num_nodes}.pickle",
                    "rb",
                ) as f:
                    pseudo_label_NRL_previous_hop = pickle.load(f)

                for sample_index in tqdm(range(len(pseudo_label_NRL_previous_hop))):
                    in_neighbors_previous_hop = pseudo_label_NRL_previous_hop[
                        sample_index
                    ]["{}-hop-in".format(khop - 1)]
                    out_neighbors_previous_hop = pseudo_label_NRL_previous_hop[
                        sample_index
                    ]["{}-hop-out".format(khop - 1)]

                    pseudo_label_NRL[sample_index] = (
                        get_pseudo_label_NRL_for_one_segment(
                            cfg,
                            khop,
                            in_neighbors_previous_hop,
                            out_neighbors_previous_hop,
                            pkg,
                            pkg_tr,
                        )
                    )
                    pseudo_label_NRL[sample_index].update(
                        pseudo_label_NRL_previous_hop[sample_index]
                    )

            # save
            with open(sample_pseudo_label_savepath, "wb") as f:
                pickle.dump(pseudo_label_NRL, f)
