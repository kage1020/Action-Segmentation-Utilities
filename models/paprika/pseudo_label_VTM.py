import pickle
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np
import polars as pl
from trainer import PaprikaConfig
from .get_samples import get_samples
from .get_nodes import get_nodes
from .helper import find_matching_of_a_segment_given_sorted_val_corres_idx
from collections import defaultdict


def obtain_document_step_task_occurrence(cfg: PaprikaConfig):
    with open(Path(cfg.document_dir) / "step_label_text.json", "r") as f:
        document = json.load(f)

    step_id = 0
    step_id_to_article_po = defaultdict(tuple)
    for article_id in range(len(document)):
        for article_step_idx in range(len(document[article_id])):
            step_id_to_article_po[step_id] = (article_id, article_step_idx)
            step_id += 1

    with open(Path(cfg.document_dir) / "article_id_to_title.txt", "r") as f:
        article_id_to_wikhow_taskname = {
            int(line.rstrip().split("\t")[0]): line.rstrip().split("\t")[1]
            for line in f.readlines()
        }

    document_tasknames = set(article_id_to_wikhow_taskname.values())
    document_taskname_to_taskid = dict()
    document_taskid_to_taskname = dict()
    for task_name in document_tasknames:
        document_taskname_to_taskid[task_name] = len(document_taskname_to_taskid)
        document_taskid_to_taskname[document_taskname_to_taskid[task_name]] = task_name

    document_step_task_occurrence = np.zeros(
        (len(step_id_to_article_po), len(document_tasknames))
    )
    for step_id in range(len(step_id_to_article_po)):
        article_id, _ = step_id_to_article_po[step_id]
        document_step_task_occurrence[
            step_id,
            document_taskname_to_taskid[article_id_to_wikhow_taskname[article_id]],
        ] += 1
    return (
        document_step_task_occurrence,
        document_taskid_to_taskname,
        document_taskname_to_taskid,
    )


def obtain_dataset_step_task_occurrence(
    cfg: PaprikaConfig, node2step, step2node, pseudo_label_VNM, samples_reverse
):
    video_meta_csv = pl.read_csv(cfg.video_meta_csv_path)
    task_ids_csv = pl.read_csv(cfg.task_id_to_task_name_csv_path, separator="\t", has_header=False)

    task_id_to_task_name_original_map = dict()
    for row in task_ids_csv.iter_rows():
        task_id = row[0]
        task_name = row[1]
        task_id_to_task_name_original_map[task_id] = task_name

    video_id_to_task_name = dict()
    for row in tqdm(video_meta_csv.iter_rows()):
        video_id = row[0]
        task_id = row[1]
        video_id_to_task_name[video_id] = task_id_to_task_name_original_map[task_id]

    task_names = set()
    for video_sid, segment_iid in tqdm(samples_reverse):
        VNM_matched_nodes = pseudo_label_VNM[samples_reverse[(video_sid, segment_iid)]][
            "indices"
        ]
        for node_id in VNM_matched_nodes:
            step_ids_this_node = node2step[node_id]
            for step_id in step_ids_this_node:
                task_name = video_id_to_task_name[video_sid]
                task_names.add(task_name)

    assert len(task_names) == len(
        task_names.intersection(set(video_id_to_task_name.values()))
    )

    dataset_taskid_to_taskname = dict()
    dataset_taskname_to_taskid = dict()
    for task_name in task_names:
        dataset_taskname_to_taskid[task_name] = len(dataset_taskname_to_taskid)
        dataset_taskid_to_taskname[dataset_taskname_to_taskid[task_name]] = task_name

    dataset_step_task_occurrence = np.zeros((len(step2node), len(task_names)))
    for video_sid, segment_iid in tqdm(samples_reverse):
        VNM_matched_nodes = pseudo_label_VNM[samples_reverse[(video_sid, segment_iid)]][
            "indices"
        ]
        for node_id in VNM_matched_nodes:
            step_ids_this_node = node2step[node_id]
            for step_id in step_ids_this_node:
                dataset_step_task_occurrence[
                    step_id,
                    dataset_taskname_to_taskid[video_id_to_task_name[video_sid]],
                ] += 1
    return (
        dataset_step_task_occurrence,
        dataset_taskid_to_taskname,
        dataset_taskname_to_taskid,
    )


def get_pseudo_label_VTM_for_one_segment(
    cfg: PaprikaConfig,
    node2step,
    VNM_matched_nodes,
    document_step_task_occurrence,
    dataset_step_task_occurrence,
):
    document_tasks_this_segment = dict()
    dataset_tasks_this_segment = dict()
    for node_id in VNM_matched_nodes:
        step_ids_this_node = node2step[node_id]
        for step_id in step_ids_this_node:
            document_taskids = np.where(document_step_task_occurrence[step_id] > 0)[0]
            for task_id in document_taskids:
                if task_id not in document_tasks_this_segment:
                    document_tasks_this_segment[task_id] = (
                        document_step_task_occurrence[step_id, task_id]
                    )
                else:
                    document_tasks_this_segment[task_id] = max(
                        document_tasks_this_segment[task_id],
                        document_step_task_occurrence[step_id, task_id],
                    )

            dataset_taskids = np.where(dataset_step_task_occurrence[step_id] > 0)[0]
            for task_id in dataset_taskids:
                if task_id not in dataset_tasks_this_segment:
                    dataset_tasks_this_segment[task_id] = dataset_step_task_occurrence[
                        step_id, task_id
                    ]
                else:
                    dataset_tasks_this_segment[task_id] = max(
                        dataset_tasks_this_segment[task_id],
                        dataset_step_task_occurrence[step_id, task_id],
                    )

    dataset_task_scores_sorted = sorted(
        dataset_tasks_this_segment.items(), key=lambda item: item[1], reverse=True
    )
    # [(task_id, task_score), ... , (task_id, task_score)]

    results = dict()
    results["document_tasks"] = list(document_tasks_this_segment.keys())
    matched_tasks, matched_tasks_scores = (
        find_matching_of_a_segment_given_sorted_val_corres_idx(
            [task_score for (task_id, task_score) in dataset_task_scores_sorted],
            [task_id for (task_id, task_score) in dataset_task_scores_sorted],
            criteria=cfg.label_find_tasks_criteria,
            threshold=cfg.label_find_tasks_thresh,
            topK=cfg.label_find_tasks_topK,
        )
    )
    results["dataset_tasks"] = {
        "indices": matched_tasks,
        "values": matched_tasks_scores,
    }
    return results


def get_pseudo_label_VTM(cfg: PaprikaConfig):
    sample_dir = Path(cfg.dataset.base_dir) / cfg.dataset.name / "samples"
    if not (sample_dir / "samples.pickle").exists():
        get_samples(cfg)
    with open(sample_dir / "samples.pickle", "rb") as f:
        samples = pickle.load(f)
    with open(sample_dir / "samples_reverse.pickle", "rb") as f:
        samples_reverse = pickle.load(f)

    node2step, step2node = get_nodes(cfg)

    sample_pseudo_label_savedir = (
        Path(cfg.dataset.base_dir) / cfg.dataset.name / "pseudo_labels"
    )
    sample_pseudo_label_savedir.mkdir(parents=True, exist_ok=True)

    sample_pseudo_label_savepath = (
        Path(sample_pseudo_label_savedir)
        / f"VTM-criteria_{cfg.label_find_tasks_criteria}-threshold_{cfg.label_find_tasks_thresh}-topK_{cfg.label_find_tasks_topK}.pickle"
    )

    # load VNM pseudo label file
    with open(
        Path(sample_pseudo_label_savedir)
        / f"VNM-criteria_{cfg.label_find_matched_nodes_criteria}-threshold_{cfg.label_find_matched_nodes_for_segments_thresh}-topK_{cfg.label_find_matched_nodes_for_segments_topK}-size_{cfg.num_nodes}.pickle",
        "rb",
    ) as f:
        pseudo_label_VNM = pickle.load(f)

    # obtain document_step_task_occurrence
    if not (
        sample_pseudo_label_savedir / "VTM-dataset_step_task_occurrence.pickle"
    ).exists():
        (
            document_step_task_occurrence,
            document_taskid_to_taskname,
            document_taskname_to_taskid,
        ) = obtain_document_step_task_occurrence(cfg)

        with open(
            sample_pseudo_label_savedir / "VTM-dataset_step_task_occurrence.pickle",
            "wb",
        ) as f:
            pickle.dump(document_step_task_occurrence, f)
        with open(
            sample_pseudo_label_savedir / "VTM-document_taskid_to_taskname.pickle",
            "wb",
        ) as f:
            pickle.dump(document_taskid_to_taskname, f)
        with open(
            sample_pseudo_label_savedir / "VTM-document_taskname_to_taskid.pickle",
            "wb",
        ) as f:
            pickle.dump(document_taskname_to_taskid, f)
    else:
        with open(
            sample_pseudo_label_savedir / "VTM-dataset_step_task_occurrence.pickle",
            "rb",
        ) as f:
            document_step_task_occurrence = pickle.load(f)

    # obtain dataset_step_task_occurrence
    if not (
        sample_pseudo_label_savedir / "VTM-dataset_step_task_occurrence.pickle"
    ).exists():
        (
            dataset_step_task_occurrence,
            dataset_taskid_to_taskname,
            dataset_taskname_to_taskid,
        ) = obtain_dataset_step_task_occurrence(
            cfg, node2step, step2node, pseudo_label_VNM, samples_reverse
        )

        with open(
            sample_pseudo_label_savedir / "VTM-dataset_step_task_occurrence.pickle",
            "wb",
        ) as f:
            pickle.dump(dataset_step_task_occurrence, f)
        with open(
            sample_pseudo_label_savedir / "VTM-dataset_taskid_to_taskname.pickle",
            "wb",
        ) as f:
            pickle.dump(dataset_taskid_to_taskname, f)
        with open(
            sample_pseudo_label_savedir / "VTM-dataset_taskname_to_taskid.pickle",
            "wb",
        ) as f:
            pickle.dump(dataset_taskname_to_taskid, f)
    else:
        with open(
            sample_pseudo_label_savedir / "VTM-dataset_step_task_occurrence.pickle",
            "rb",
        ) as f:
            dataset_step_task_occurrence = pickle.load(f)

    if not sample_pseudo_label_savepath.exists():
        # start processing
        pseudo_label_VTM = dict()
        for sample_index in tqdm(range(len(samples))):
            VNM_matched_nodes = pseudo_label_VNM[sample_index]["indices"]

            pseudo_label_VTM[sample_index] = get_pseudo_label_VTM_for_one_segment(
                cfg,
                node2step,
                VNM_matched_nodes,
                document_step_task_occurrence,
                dataset_step_task_occurrence,
            )

        # save
        with open(sample_pseudo_label_savepath, "wb") as f:
            pickle.dump(pseudo_label_VTM, f)
