import pickle
import random
from pathlib import Path
from tqdm import tqdm

from trainer import PaprikaConfig


def partition_dataset(cfg: PaprikaConfig):
    """Randomly partition the dataset's features and pseudo labels."""

    with open(
        Path(cfg.dataset.base_dir)
        / cfg.dataset.name
        / "features"
        / "feats_all-mean_agg.pickle",
        "rb",
    ) as f:
        feats_all = pickle.load(f)

    all_indices = list(feats_all.keys())
    num_each_partition = len(all_indices) // cfg.num_partitions
    num_allsamples = num_each_partition * cfg.num_partitions

    random.shuffle(all_indices)
    all_indices = all_indices[:num_allsamples]

    # VNM
    sample_pseudo_label_savepath = (
        Path(cfg.dataset.base_dir)
        / cfg.dataset.name
        / "pseudo_labels"
        / f"VNM-criteria_{cfg.label_find_matched_nodes_criteria}-threshold_{cfg.label_find_matched_nodes_for_segments_thresh}-topK_{cfg.label_find_matched_nodes_for_segments_topK}-size_{cfg.num_nodes}.pickle"
    )
    with open(sample_pseudo_label_savepath, "rb") as f:
        pseudo_labels_all_VNM = pickle.load(f)

    # VTM
    sample_pseudo_label_savepath = (
        Path(cfg.dataset.base_dir)
        / cfg.dataset.name
        / "pseudo_labels"
        / f"VTM-criteria_{cfg.label_find_tasks_criteria}-threshold_{cfg.label_find_tasks_thresh}-topK_{cfg.label_find_tasks_topK}.pickle"
    )
    with open(sample_pseudo_label_savepath, "rb") as f:
        pseudo_labels_all_VTM = pickle.load(f)

    # TCL
    sample_pseudo_label_savepath = (
        Path(cfg.dataset.base_dir)
        / cfg.dataset.name
        / "pseudo_labels"
        / f"TCL-criteria_{cfg.label_find_tasks_criteria}-threshold_{cfg.label_find_tasks_thresh}-topK_{cfg.label_find_tasks_topK}.pickle"
    )
    with open(sample_pseudo_label_savepath, "rb") as f:
        pseudo_labels_all_TCL = pickle.load(f)

    # NRL
    sample_pseudo_label_savepath = (
        Path(cfg.dataset.base_dir)
        / cfg.dataset.name
        / "pseudo_labels"
        / f"NRL-hop_{cfg.label_khop}-criteria_{cfg.label_find_neighbors_criteria}-threshold_{cfg.label_find_neighbors_thresh}-topK_{cfg.label_find_neighbors_topK}-size_{cfg.num_nodes}.pickle"
    )
    with open(sample_pseudo_label_savepath, "rb") as f:
        pseudo_labels_all_NRL = pickle.load(f)

    # Start partitioning
    for i in range(cfg.num_partitions):
        index_keys_this_partition = all_indices[
            i * num_each_partition : (i + 1) * num_each_partition
        ]

        feats_this = dict()
        pseudo_labels_VNM_this = dict()
        pseudo_labels_VTM_this = dict()
        pseudo_labels_TCL_this = dict()
        pseudo_labels_NRL_this = dict()

        for key in tqdm(index_keys_this_partition):
            feats_this[key] = feats_all[key]
            pseudo_labels_VNM_this[key] = pseudo_labels_all_VNM[key]
            pseudo_labels_VTM_this[key] = pseudo_labels_all_VTM[key]
            pseudo_labels_TCL_this[key] = pseudo_labels_all_TCL[key]
            pseudo_labels_NRL_this[key] = pseudo_labels_all_NRL[key]

        with open(
            Path(cfg.dataset.base_dir)
            / cfg.dataset.name
            / f"index_keys_this_partition-rank_{i}-of-{cfg.num_partitions}.pickle",
            "wb",
        ) as f:
            pickle.dump(index_keys_this_partition, f)

        with open(
            Path(cfg.dataset.base_dir)
            / cfg.dataset.name
            / "features"
            / f"feats_all-mean_agg-rank_{i}-of-{cfg.num_partitions}.pickle",
            "wb",
        ) as f:
            pickle.dump(feats_this, f)

        sample_pseudo_label_savepath = (
            Path(cfg.dataset.base_dir)
            / cfg.dataset.name
            / "pseudo_labels"
            / f"VNM-criteria_{cfg.label_find_matched_nodes_criteria}-threshold_{cfg.label_find_matched_nodes_for_segments_thresh}-topK_{cfg.label_find_matched_nodes_for_segments_topK}-size_{cfg.num_nodes}-rank_{i}-of-{cfg.num_partitions}.pickle"
        )
        with open(sample_pseudo_label_savepath, "wb") as f:
            pickle.dump(pseudo_labels_VNM_this, f)

        sample_pseudo_label_savepath = (
            Path(cfg.dataset.base_dir)
            / cfg.dataset.name
            / "pseudo_labels"
            / f"VTM-criteria_{cfg.label_find_tasks_criteria}-threshold_{cfg.label_find_tasks_thresh}-topK_{cfg.label_find_tasks_topK}-rank_{i}-of-{cfg.num_partitions}.pickle"
        )
        with open(sample_pseudo_label_savepath, "wb") as f:
            pickle.dump(pseudo_labels_VTM_this, f)

        sample_pseudo_label_savepath = (
            Path(cfg.dataset.base_dir)
            / cfg.dataset.name
            / "pseudo_labels"
            / f"TCL-criteria_{cfg.label_find_tasks_criteria}-threshold_{cfg.label_find_tasks_thresh}-topK_{cfg.label_find_tasks_topK}-rank_{i}-of-{cfg.num_partitions}.pickle"
        )
        with open(sample_pseudo_label_savepath, "wb") as f:
            pickle.dump(pseudo_labels_TCL_this, f)

        sample_pseudo_label_savepath = (
            Path(cfg.dataset.base_dir)
            / cfg.dataset.name
            / "pseudo_labels"
            / f"NRL-hop_{cfg.label_khop}-criteria_{cfg.label_find_neighbors_criteria}-threshold_{cfg.label_find_neighbors_thresh}-topK_{cfg.label_find_neighbors_topK}-size_{cfg.num_nodes}-rank_{i}-of-{cfg.num_partitions}.pickle"
        )
        with open(sample_pseudo_label_savepath, "wb") as f:
            pickle.dump(pseudo_labels_NRL_this, f)
