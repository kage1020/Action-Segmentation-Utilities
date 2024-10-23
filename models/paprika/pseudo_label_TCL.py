import pickle
from tqdm import tqdm
from pathlib import Path
import numpy as np

from trainer import PaprikaConfig
from .get_samples import get_samples
from .get_nodes import get_nodes
from .helper import find_matching_of_a_segment_given_sorted_val_corres_idx


def get_pseudo_label_TCL_for_one_segment(
    cfg: PaprikaConfig,
    step2node,
    VTM_matched_wikihow_tasks,
    VTM_matched_howto100m_tasks,
    wikihow_step_task_occurrence,
    howto100m_step_task_occurrence,
):
    wikihow_tasknodes_this_segment = dict()
    for task_id in VTM_matched_wikihow_tasks:
        wikihow_stepids = np.where(wikihow_step_task_occurrence[:, task_id] > 0)[0]
        for step_id in wikihow_stepids:
            node_id = step2node[step_id]
            if node_id not in wikihow_tasknodes_this_segment:
                wikihow_tasknodes_this_segment[node_id] = wikihow_step_task_occurrence[
                    step_id, task_id
                ]
            else:
                wikihow_tasknodes_this_segment[node_id] = max(
                    wikihow_tasknodes_this_segment[node_id],
                    wikihow_step_task_occurrence[step_id, task_id],
                )

    howto100m_tasknodes_this_segment = dict()
    for task_id in VTM_matched_howto100m_tasks:
        howto100m_stepids = np.where(howto100m_step_task_occurrence[:, task_id] > 0)[0]
        for step_id in howto100m_stepids:
            node_id = step2node[step_id]
            if node_id not in howto100m_tasknodes_this_segment:
                howto100m_tasknodes_this_segment[node_id] = (
                    howto100m_step_task_occurrence[step_id, task_id]
                )
            else:
                howto100m_tasknodes_this_segment[node_id] = max(
                    howto100m_tasknodes_this_segment[node_id],
                    howto100m_step_task_occurrence[step_id, task_id],
                )

    howto100m_tasknodes_scores_sorted = sorted(
        howto100m_tasknodes_this_segment.items(), key=lambda item: item[1], reverse=True
    )
    # [(node_id, node_score), ... , (node_id, node_score)]

    results = dict()
    results["wikihow_tasknodes"] = list(wikihow_tasknodes_this_segment.keys())
    matched_tasknodes, matched_tasknodes_scores = (
        find_matching_of_a_segment_given_sorted_val_corres_idx(
            [node_score for (node_id, node_score) in howto100m_tasknodes_scores_sorted],
            [node_id for (node_id, node_score) in howto100m_tasknodes_scores_sorted],
            criteria=cfg.label_find_tasknodes_criteria,
            threshold=cfg.label_find_tasknodes_thresh,
            topK=cfg.label_find_tasknodes_topK,
        )
    )
    results["howto100m_tasknodes"] = {
        "indices": matched_tasknodes,
        "values": matched_tasknodes_scores,
    }
    return results


def get_pseudo_label_TCL(cfg: PaprikaConfig):
    sample_dir = Path(cfg.dataset.base_dir) / cfg.dataset.name / "samples"
    if not (sample_dir / "samples.pickle").exists():
        get_samples(cfg)
    with open(sample_dir / "samples.pickle", "rb") as f:
        samples = pickle.load(f)

    _, step2node = get_nodes(cfg)

    sample_pseudo_label_savedir = (
        Path(cfg.dataset.base_dir) / cfg.dataset.name / "pseudo_labels"
    )
    sample_pseudo_label_savedir.mkdir(parents=True, exist_ok=True)

    sample_pseudo_label_savepath = (
        sample_pseudo_label_savedir
        / f"TCL-criteria_{cfg.label_find_tasknodes_criteria}-threshold_{cfg.label_find_tasknodes_thresh}-topK_{cfg.label_find_tasknodes_topK}.pickle"
    )

    with open(
        sample_pseudo_label_savedir / "VTM-wikihow_step_task_occurrence.pickle",
        "rb",
    ) as f:
        wikihow_step_task_occurrence = pickle.load(f)
    with open(
        sample_pseudo_label_savedir / "VTM-howto100m_step_task_occurrence.pickle",
        "rb",
    ) as f:
        howto100m_step_task_occurrence = pickle.load(f)

    # load VTM pseudo label file
    with open(
        sample_pseudo_label_savedir
        / f"VTM-criteria_{cfg.label_find_tasks_criteria}-threshold_{cfg.label_find_tasks_thresh}-topK_{cfg.label_find_tasks_topK}.pickle",
        "rb",
    ) as f:
        pseudo_label_VTM = pickle.load(f)

    if sample_pseudo_label_savepath.exists():
        # start processing
        pseudo_label_TCL = dict()
        for sample_index in tqdm(range(len(samples))):
            VTM_matched_wikihow_tasks = pseudo_label_VTM[sample_index]["wikihow_tasks"]
            VTM_matched_howto100m_tasks = pseudo_label_VTM[sample_index][
                "howto100m_tasks"
            ]["indices"][: cfg.label_num_howto100m_tasks_to_consider]

            pseudo_label_TCL[sample_index] = get_pseudo_label_TCL_for_one_segment(
                cfg,
                step2node,
                VTM_matched_wikihow_tasks,
                VTM_matched_howto100m_tasks,
                wikihow_step_task_occurrence,
                howto100m_step_task_occurrence,
            )

        # save
        with open(sample_pseudo_label_savepath, "wb") as f:
            pickle.dump(pseudo_label_TCL, f)
