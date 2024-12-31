import pickle
import json
import random
from pathlib import Path
from collections import defaultdict
import polars as pl
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix
import torch
import torch.nn.functional as F

from base.main import Base
from configs.paprika import PaprikaConfig, PaprikaPseudoConfig, PaprikaDownstreamConfig


class Builder(Base):
    def __init__(self, cfg: PaprikaConfig | PaprikaPseudoConfig | PaprikaDownstreamConfig):
        self.cfg = cfg

    def get_all_video_ids(self) -> list[str]:
        video_ids = (
            Path(self.cfg.dataset.base_dir) / self.cfg.dataset.name / "video_IDs.npy"
        )
        if video_ids.exists():
            videos = np.load(video_ids)
        else:
            videos = []
            video_features_path = (
                Path(self.cfg.dataset.base_dir) / self.cfg.dataset.name / self.cfg.dataset.feature_dir
            )
            video_features = list(video_features_path.glob("*"))
            assert len(video_features) > 0, "No video features found!"

            for f in tqdm(video_features, desc="Loading videos", leave=False):
                videos.append(Path(f).stem)
            np.save(video_ids, videos)

        return videos

    def get_step_des_feats(self, language_model="MPNet") -> np.ndarray:
        """
        Returns:
            step_des_feats: np.ndarray of shape (num_steps, embedding_dim)
        """
        if language_model == "MPNet":
            step_des_feats = np.load(
                Path(f"{self.cfg.dataset.base_dir}/{self.cfg.document.name}")
                / "mpnet_feat.npy"
            )

        elif language_model == "S3D":
            with open(
                Path(f"{self.cfg.dataset.base_dir}/{self.cfg.document.name}")
                / "s3d_text_feat"
                / "step_embeddings.pickle",
                "rb",
            ) as f:
                step_des_feats = pickle.load(f)
        else:
            raise ValueError(
                f"Language model {language_model} is not supported!\nPlease choose from ['MPNet', 'S3D']"
            )

        return step_des_feats

    def find_matching_of_a_segment_given_sorted_val_corres_idx(
        self, sorted_values, sorted_indices, criteria="threshold", threshold=0.7, topK=3
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
            raise NotImplementedError(
                f"The criteria is not implemented!\nFunc: {__name__}\nFile: {__file__}"
            )

        return matched_steps, matched_steps_score

    def find_matching_of_a_segment(
        self, sim_scores, criteria="threshold", threshold=0.7, topK=3
    ):
        sorted_values = np.sort(sim_scores)[::-1]  # sort in descending order
        sorted_indices = np.argsort(
            -sim_scores
        )  # indices of sorting in descending order

        matched_steps, matched_steps_score = (
            self.find_matching_of_a_segment_given_sorted_val_corres_idx(
                sorted_values,
                sorted_indices,
                criteria=criteria,
                threshold=threshold,
                topK=topK,
            )
        )

        return matched_steps, matched_steps_score

    def gather_all_frame_S3D_embeddings(self):
        videos = self.get_all_video_ids()

        frame_embeddings = []
        frame_lookup_table = []

        videos_missing_features = set()
        for v in tqdm(videos, desc="Loading video features", leave=False):
            try:
                video_s3d: np.ndarray = np.load(
                    Path(self.cfg.dataset.base_dir)
                    / self.cfg.dataset.name
                    / self.cfg.dataset.feature_dir
                    / f"{v}.npy"
                )
                # video_s3d shape: (num_clips, num_subclips, 512)

                for c_idx in range(video_s3d.shape[0]):
                    frame_embeddings.append(
                        np.float64(np.mean(video_s3d[c_idx], axis=0))
                    )
                    frame_lookup_table.append((v, c_idx))

            except FileNotFoundError:
                videos_missing_features.add(v)

        if len(videos_missing_features) > 0:
            with open("videos_missing_features.pickle", "wb") as f:
                pickle.dump(videos_missing_features, f)
        assert len(videos_missing_features) == 0, (
            "There are videos missing features! "
            + "Please check saved videos_missing_features.pickle."
        )

        frame_embeddings = np.array(frame_embeddings)

        # segment video embeddings shape: (3741608, 512) for the subset
        # segment video embeddings shape: (51053844, 512) for the fullset
        return frame_embeddings, frame_lookup_table

    def find_step_similarities_for_segments_using_frame(
        self,
        step_des_feats,
        segment_video_embeddings,
        segment_video_lookup_table,
    ):
        for segment_id in tqdm(range(len(segment_video_embeddings))):
            v, cidx = segment_video_lookup_table[segment_id]
            save_path = Path(
                f"{self.cfg.dataset.base_dir}/{self.cfg.dataset.name}/sim_scores/{v}/segment_{cidx}.npy"
            )
            if not save_path.exists():
                # dot product as similarity score
                sim_scores = np.einsum(
                    "ij,ij->i",
                    step_des_feats,
                    segment_video_embeddings[segment_id][np.newaxis, ...],
                )

                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(save_path, sim_scores)

    def get_sim_scores(self):
        step_des_feats = self.get_step_des_feats("S3D")
        segment_video_embeddings, segment_video_lookup_table = (
            self.gather_all_frame_S3D_embeddings()
        )
        self.find_step_similarities_for_segments_using_frame(
            step_des_feats,
            segment_video_embeddings,
            segment_video_lookup_table,
        )

    def get_samples(self):
        videos = self.get_all_video_ids()

        samples = list()
        for v in tqdm(videos):
            video_s3d: np.ndarray = np.load(
                Path(self.cfg.dataset.base_dir)
                / self.cfg.dataset.name
                / "features"
                / f"{v}.npy"
            )

            for c_idx in range(video_s3d.shape[0]):
                samples.append((v, c_idx))
        random.shuffle(samples)

        samples_id2name = dict()
        samples_name2id = dict()
        for i in range(len(samples)):
            samples_id2name[i] = samples[i]
            samples_name2id[samples[i]] = i

        sample_dir = Path(self.cfg.dataset.base_dir) / self.cfg.dataset.name / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        with open(sample_dir / "samples.pickle", "wb") as f:
            pickle.dump(samples_id2name, f)
        with open(sample_dir / "samples_reverse.pickle", "wb") as f:
            pickle.dump(samples_name2id, f)
        return

    def get_nodes_by_removing_step_duplicates(
        self, step_des_feats=None
    ) -> tuple[dict[int, list[int]], dict[int, int]]:
        if self.cfg.remove_step_duplicates:
            node2step_path = (
                Path(f"{self.cfg.dataset.base_dir}/{self.cfg.document.name}")
                / "node2step.pickle"
            )
            step2node_path = (
                Path(f"{self.cfg.dataset.base_dir}/{self.cfg.document.name}")
                / "step2node.pickle"
            )

            if node2step_path.exists() and step2node_path.exists():
                with open(node2step_path, "rb") as f:
                    node2step = pickle.load(f)
                with open(step2node_path, "rb") as f:
                    step2node = pickle.load(f)
            else:
                assert step_des_feats is not None

                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    linkage=self.cfg.step_clustering_linkage,
                    distance_threshold=self.cfg.step_clustering_distance_threshold,
                    affinity=self.cfg.step_clustering_affinity,
                ).fit(step_des_feats)
                # distance_threshold:
                #   The linkage distance threshold above which, clusters will not be merged.
                num_nodes = clustering.n_clusters_

                node2step, step2node = dict(), dict()
                for cluster_id in range(num_nodes):
                    cluster_members = np.where(clustering.labels_ == cluster_id)[0]
                    node2step[cluster_id] = cluster_members
                    for step_id in cluster_members:
                        step2node[step_id] = cluster_id
                with open(node2step_path, "wb") as f:
                    pickle.dump(node2step, f)
                with open(step2node_path, "wb") as f:
                    pickle.dump(step2node, f)

        else:
            node2step = {i: [i] for i in range(self.cfg.num_nodes)}
            step2node = {i: i for i in range(self.cfg.num_nodes)}

        return node2step, step2node

    def get_nodes(self):
        step_des_feats = self.get_step_des_feats("MPNet")
        node2step, step2node = self.get_nodes_by_removing_step_duplicates(
            step_des_feats
        )
        return node2step, step2node

    def get_edges_between_document_steps_in_document(self):
        with open(
            Path(f"{self.cfg.dataset.base_dir}/{self.cfg.document.name}")
            / "step_label_text.json",
            "r",
        ) as f:
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

    def threshold_and_normalize(self, G: np.ndarray, edge_min_aggconf=1000):
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
                    G_new[i, j] = (np.log(G[i, j]) - 0) / (
                        max_val - 0
                    )  # log min max norm
        G = G_new
        return G

    def get_edges_between_document_steps_in_dataset(self, total_num_steps):
        sim_score_path = (
            Path(self.cfg.dataset.base_dir) / self.cfg.dataset.name / "sim_scores"
        )

        videos = self.get_all_video_ids()

        edges_metas = list()
        for video in videos:
            edges_metas.append(
                self.get_edges_between_document_steps_of_one_dataset_video(
                    video, sim_score_path
                )
            )

        dataset_steps_1hop_edges = np.zeros((total_num_steps, total_num_steps))
        for edges_meta in edges_metas:
            for [predecessor, successor, confidence] in edges_meta:
                dataset_steps_1hop_edges[predecessor, successor] += confidence

        dataset_steps_1hop_edges = self.threshold_and_normalize(
            dataset_steps_1hop_edges, self.cfg.edge_min_aggconf
        )

        return dataset_steps_1hop_edges

    def get_edges_between_document_steps_of_one_dataset_video(
        self, video: str, sim_score_path: str
    ):
        sim_score_paths_of_segments_this_video = sorted(
            Path(f"{sim_score_path}/{video}").glob("segment_*.npy")
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

            predecessors, _ = self.find_matching_of_a_segment(
                segment_pre_sim_scores,
                criteria=self.cfg.graph_find_matched_steps_criteria,
                threshold=self.cfg.graph_find_matched_steps_for_segments_threshold,
                topK=self.cfg.graph_find_matched_steps_for_segments_topK,
            )

            successors, _ = self.find_matching_of_a_segment(
                segment_suc_sim_scores,
                criteria=self.cfg.graph_find_matched_steps_criteria,
                threshold=self.cfg.graph_find_matched_steps_for_segments_threshold,
                topK=self.cfg.graph_find_matched_steps_for_segments_topK,
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

    def keep_highest_conf_for_each_candidate(self, candidates):
        edges: dict[tuple[int, int], float] = dict()
        for node_id, direct_outnode_id in tqdm(candidates):
            max_conf = np.max(candidates[(node_id, direct_outnode_id)])

            edges[(node_id, direct_outnode_id)] = max_conf
        return edges

    def build_pkg_adj_matrix(self, edges, num_nodes):
        pkg = np.zeros((num_nodes, num_nodes))
        for node_id, direct_outnode_id in tqdm(edges):
            pkg[node_id, direct_outnode_id] = edges[(node_id, direct_outnode_id)]
        return pkg

    def get_node_transition_candidates(
        self, step2node: dict[int, int], G_wikihow: csr_matrix, G_howto100m: csr_matrix
    ):
        candidates: dict[tuple[int, int], list[float]] = dict(list)

        for step_id in tqdm(range(len(step2node))):
            for direct_outstep_id in G_wikihow[step_id].indices:
                conf: float = G_wikihow[step_id, direct_outstep_id]
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

    def get_edges(self):
        # --  get the edges between step headlines
        G_document = self.get_edges_between_document_steps_in_document()
        G_dataset = self.get_edges_between_document_steps_in_dataset(
            G_document.shape[0]
        )
        G_document_csr, G_howto100m_csr = csr_matrix(G_document), csr_matrix(G_dataset)

        # -- turn edges between step headlines into edges between nodes
        node2step, step2node = self.get_nodes()
        node_transition_candidates = self.get_node_transition_candidates(
            step2node, G_document_csr, G_howto100m_csr
        )
        pkg_edges = self.keep_highest_conf_for_each_candidate(
            node_transition_candidates
        )
        pkg = self.build_pkg_adj_matrix(pkg_edges, len(node2step))

        return pkg, G_document, G_dataset

    def get_pseudo_label_VNM_for_one_segment(
        self, node2step, step2node, sample_gt_path
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

        matched_nodes, matched_nodes_scores = self.find_matching_of_a_segment(
            node_scores_arr,
            criteria=self.cfg.label_find_matched_nodes_criteria,
            threshold=self.cfg.label_find_matched_nodes_for_segments_threshold,
            topK=self.cfg.label_find_matched_nodes_for_segments_topK,
        )

        pseudo_label_VNM = {"indices": matched_nodes, "values": matched_nodes_scores}
        return pseudo_label_VNM

    def get_pseudo_label_VNM(self):
        sample_path = (
            Path(self.cfg.dataset.base_dir)
            / self.cfg.dataset.name
            / "samples"
            / "samples.pickle"
        )
        if not sample_path.exists():
            self.get_samples()
        with open(sample_path, "rb") as f:
            samples = pickle.load(f)

        node2step, step2node = self.get_nodes()

        sim_score_path = (
            Path(self.cfg.dataset.base_dir) / self.cfg.dataset.name / "sim_scores"
        )
        sample_pseudo_label_savedir = (
            Path(self.cfg.dataset.base_dir) / self.cfg.dataset.name / "pseudo_labels"
        )
        sample_pseudo_label_savedir.mkdir(parents=True, exist_ok=True)

        sample_pseudo_label_savepath = (
            Path(sample_pseudo_label_savedir)
            / f"VNM-criteria_{self.cfg.label_find_matched_nodes_criteria}-threshold_{self.cfg.label_find_matched_nodes_for_segments_threshold}-topK_{self.cfg.label_find_matched_nodes_for_segments_topK}-size_{self.cfg.num_nodes}.pickle"
        )

        if not sample_pseudo_label_savepath.exists():
            # start processing
            pseudo_label_VNM = dict()
            for sample_index in tqdm(range(len(samples))):
                (video_sid, segment_iid) = samples[sample_index]
                segment_sid = f"segment_{segment_iid}"
                sample_gt_path = Path(sim_score_path) / video_sid / f"{segment_sid}.npy"

                pseudo_label_VNM[sample_index] = (
                    self.get_pseudo_label_VNM_for_one_segment(
                        node2step, step2node, sample_gt_path
                    )
                )

            # save
            with open(sample_pseudo_label_savepath, "wb") as f:
                pickle.dump(pseudo_label_VNM, f)

    def obtain_document_step_task_occurrence(self):
        document_dir = Path(self.cfg.dataset.base_dir) / self.cfg.document.name
        with open(document_dir / "step_label_text.json", "r") as f:
            document = json.load(f)

        step_id = 0
        step_id_to_article_po = defaultdict(tuple)
        for article_id in range(len(document)):
            for article_step_idx in range(len(document[article_id])):
                step_id_to_article_po[step_id] = (article_id, article_step_idx)
                step_id += 1

        with open(document_dir / "article_id_to_title.txt", "r") as f:
            article_id_to_wikhow_taskname = {
                int(line.rstrip().split("\t")[0]): line.rstrip().split("\t")[1]
                for line in f.readlines()
            }

        document_tasknames = set(article_id_to_wikhow_taskname.values())
        document_taskname_to_taskid = dict()
        document_taskid_to_taskname = dict()
        for task_name in document_tasknames:
            document_taskname_to_taskid[task_name] = len(document_taskname_to_taskid)
            document_taskid_to_taskname[document_taskname_to_taskid[task_name]] = (
                task_name
            )

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
        self, node2step, step2node, pseudo_label_VNM, samples_reverse
    ):
        video_meta_csv = pl.read_csv(self.cfg.video_meta_csv_path)
        task_ids_csv = pl.read_csv(
            self.cfg.task_id_to_task_name_csv_path, separator="\t", has_header=False
        )

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
            VNM_matched_nodes = pseudo_label_VNM[
                samples_reverse[(video_sid, segment_iid)]
            ]["indices"]
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
            dataset_taskid_to_taskname[dataset_taskname_to_taskid[task_name]] = (
                task_name
            )

        dataset_step_task_occurrence = np.zeros((len(step2node), len(task_names)))
        for video_sid, segment_iid in tqdm(samples_reverse):
            VNM_matched_nodes = pseudo_label_VNM[
                samples_reverse[(video_sid, segment_iid)]
            ]["indices"]
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
        self,
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
                document_taskids = np.where(document_step_task_occurrence[step_id] > 0)[
                    0
                ]
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
                        dataset_tasks_this_segment[task_id] = (
                            dataset_step_task_occurrence[step_id, task_id]
                        )
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
            self.find_matching_of_a_segment_given_sorted_val_corres_idx(
                [task_score for (task_id, task_score) in dataset_task_scores_sorted],
                [task_id for (task_id, task_score) in dataset_task_scores_sorted],
                criteria=self.cfg.label_find_tasks_criteria,
                threshold=self.cfg.label_find_tasks_threshold,
                topK=self.cfg.label_find_tasks_topK,
            )
        )
        results["dataset_tasks"] = {
            "indices": matched_tasks,
            "values": matched_tasks_scores,
        }
        return results

    def get_pseudo_label_VTM(self):
        sample_dir = Path(self.cfg.dataset.base_dir) / self.cfg.dataset.name / "samples"
        if not (sample_dir / "samples.pickle").exists():
            self.get_samples()
        with open(sample_dir / "samples.pickle", "rb") as f:
            samples = pickle.load(f)
        with open(sample_dir / "samples_reverse.pickle", "rb") as f:
            samples_reverse = pickle.load(f)

        node2step, step2node = self.get_nodes()

        sample_pseudo_label_savedir = (
            Path(self.cfg.dataset.base_dir) / self.cfg.dataset.name / "pseudo_labels"
        )
        sample_pseudo_label_savedir.mkdir(parents=True, exist_ok=True)

        sample_pseudo_label_savepath = (
            Path(sample_pseudo_label_savedir)
            / f"VTM-criteria_{self.cfg.label_find_tasks_criteria}-threshold_{self.cfg.label_find_tasks_threshold}-topK_{self.cfg.label_find_tasks_topK}.pickle"
        )

        # load VNM pseudo label file
        with open(
            Path(sample_pseudo_label_savedir)
            / f"VNM-criteria_{self.cfg.label_find_matched_nodes_criteria}-threshold_{self.cfg.label_find_matched_nodes_for_segments_threshold}-topK_{self.cfg.label_find_matched_nodes_for_segments_topK}-size_{self.cfg.num_nodes}.pickle",
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
            ) = self.obtain_document_step_task_occurrence()

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
            ) = self.obtain_dataset_step_task_occurrence(
                node2step, step2node, pseudo_label_VNM, samples_reverse
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

                pseudo_label_VTM[sample_index] = (
                    self.get_pseudo_label_VTM_for_one_segment(
                        node2step,
                        VNM_matched_nodes,
                        document_step_task_occurrence,
                        dataset_step_task_occurrence,
                    )
                )

            # save
            with open(sample_pseudo_label_savepath, "wb") as f:
                pickle.dump(pseudo_label_VTM, f)

    def get_pseudo_label_TCL_for_one_segment(
        self,
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
                    wikihow_tasknodes_this_segment[node_id] = (
                        wikihow_step_task_occurrence[step_id, task_id]
                    )
                else:
                    wikihow_tasknodes_this_segment[node_id] = max(
                        wikihow_tasknodes_this_segment[node_id],
                        wikihow_step_task_occurrence[step_id, task_id],
                    )

        howto100m_tasknodes_this_segment = dict()
        for task_id in VTM_matched_howto100m_tasks:
            howto100m_stepids = np.where(
                howto100m_step_task_occurrence[:, task_id] > 0
            )[0]
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
            howto100m_tasknodes_this_segment.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        # [(node_id, node_score), ... , (node_id, node_score)]

        results = dict()
        results["wikihow_tasknodes"] = list(wikihow_tasknodes_this_segment.keys())
        matched_tasknodes, matched_tasknodes_scores = (
            self.find_matching_of_a_segment_given_sorted_val_corres_idx(
                [
                    node_score
                    for (node_id, node_score) in howto100m_tasknodes_scores_sorted
                ],
                [
                    node_id
                    for (node_id, node_score) in howto100m_tasknodes_scores_sorted
                ],
                criteria=self.cfg.label_find_tasknodes_criteria,
                threshold=self.cfg.label_find_tasknodes_threshold,
                topK=self.cfg.label_find_tasknodes_topK,
            )
        )
        results["howto100m_tasknodes"] = {
            "indices": matched_tasknodes,
            "values": matched_tasknodes_scores,
        }
        return results

    def get_pseudo_label_TCL(self):
        sample_dir = Path(self.cfg.dataset.base_dir) / self.cfg.dataset.name / "samples"
        if not (sample_dir / "samples.pickle").exists():
            self.get_samples()
        with open(sample_dir / "samples.pickle", "rb") as f:
            samples = pickle.load(f)

        _, step2node = self.get_nodes()

        sample_pseudo_label_savedir = (
            Path(self.cfg.dataset.base_dir) / self.cfg.dataset.name / "pseudo_labels"
        )
        sample_pseudo_label_savedir.mkdir(parents=True, exist_ok=True)

        sample_pseudo_label_savepath = (
            sample_pseudo_label_savedir
            / f"TCL-criteria_{self.cfg.label_find_tasknodes_criteria}-threshold_{self.cfg.label_find_tasknodes_threshold}-topK_{self.cfg.label_find_tasknodes_topK}.pickle"
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
            / f"VTM-criteria_{self.cfg.label_find_tasks_criteria}-threshold_{self.cfg.label_find_tasks_threshold}-topK_{self.cfg.label_find_tasks_topK}.pickle",
            "rb",
        ) as f:
            pseudo_label_VTM = pickle.load(f)

        if sample_pseudo_label_savepath.exists():
            # start processing
            pseudo_label_TCL = dict()
            for sample_index in tqdm(range(len(samples))):
                VTM_matched_wikihow_tasks = pseudo_label_VTM[sample_index][
                    "wikihow_tasks"
                ]
                VTM_matched_howto100m_tasks = pseudo_label_VTM[sample_index][
                    "howto100m_tasks"
                ]["indices"][: self.cfg.label_num_dataset_tasks_to_consider]

                pseudo_label_TCL[sample_index] = (
                    self.get_pseudo_label_TCL_for_one_segment(
                        self.cfg,
                        step2node,
                        VTM_matched_wikihow_tasks,
                        VTM_matched_howto100m_tasks,
                        wikihow_step_task_occurrence,
                        howto100m_step_task_occurrence,
                    )
                )

            # save
            with open(sample_pseudo_label_savepath, "wb") as f:
                pickle.dump(pseudo_label_TCL, f)

    def get_pseudo_label_NRL_for_one_segment(
        self,
        khop,
        in_neighbors_previous_hop,
        out_neighbors_previous_hop,
        pkg: csr_matrix,
        pkg_tr: csr_matrix,
    ):
        results: dict[str, dict[str, list[int]]] = dict()
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
                            neighbors_previous_hop["values"][i]
                            * G[nei_prehop, direct_nei]
                        )
                    else:
                        node_scores[direct_nei] = G[nei_prehop, direct_nei]
            node_scores_sorted = sorted(
                node_scores.items(), key=lambda item: item[1], reverse=True
            )
            # [(node_id, node_score), ... , (node_id, node_score)]

            (matched_neihgbor_nodes, matched_neihgbor_nodes_scores) = (
                self.find_matching_of_a_segment_given_sorted_val_corres_idx(
                    [node_score for (node_id, node_score) in node_scores_sorted],
                    [node_id for (node_id, node_score) in node_scores_sorted],
                    criteria=self.cfg.label_find_neighbors_criteria,
                    threshold=self.cfg.label_find_neighbors_threshold,
                    topK=self.cfg.label_find_neighbors_topK,
                )
            )
            results["{}-hop-{}".format(khop, direction_key)] = {
                "indices": matched_neihgbor_nodes,
                "values": matched_neihgbor_nodes_scores,
            }
        return results

    def get_pseudo_label_NRL(self):
        pkg, _, _ = self.get_edges()

        pkg_tr = np.transpose(pkg)
        pkg, pkg_tr = csr_matrix(pkg), csr_matrix(pkg_tr)

        sample_pseudo_label_savedir = (
            Path(self.cfg.dataset.base_dir) / self.cfg.dataset.name / "pseudo_labels"
        )
        sample_pseudo_label_savedir.mkdir(parents=True, exist_ok=True)

        for khop in range(1, self.cfg.label_khop + 1):
            sample_pseudo_label_savepath = (
                Path(sample_pseudo_label_savedir)
                / f"NRL-hop_{khop}-criteria_{self.cfg.label_find_neighbors_criteria}-threshold_{self.cfg.label_find_neighbors_threshold}-topK_{self.cfg.label_find_neighbors_topK}-size_{self.cfg.num_nodes}.pickle"
            )
            if not sample_pseudo_label_savepath.exists():
                # start processing
                pseudo_label_NRL: dict[int, dict[str, dict[str, list[int]]]] = dict()
                if khop == 1:
                    with open(
                        sample_pseudo_label_savedir
                        / f"VNM-criteria_{self.cfg.label_find_matched_nodes_criteria}-threshold_{self.cfg.label_find_matched_nodes_for_segments_threshold}-topK_{self.cfg.label_find_matched_nodes_for_segments_topK}-size_{self.cfg.num_nodes}.pickle",
                        "rb",
                    ) as f:
                        pseudo_label_VNM = pickle.load(f)

                    for sample_index in tqdm(range(len(pseudo_label_VNM))):
                        pseudo_label_NRL[sample_index] = (
                            self.get_pseudo_label_NRL_for_one_segment(
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
                        / f"NRL-hop_{khop - 1}-criteria_{self.cfg.label_find_neighbors_criteria}-threshold_{self.cfg.label_find_neighbors_threshold}-topK_{self.cfg.label_find_neighbors_topK}-size_{self.cfg.num_nodes}.pickle",
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
                            self.get_pseudo_label_NRL_for_one_segment(
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


    def gather_all_narration_MPNet_embeddings(self):
        videos = self.get_all_video_ids()

        narration_embeddings = []
        narration_lookup_table = []
        videos_missing_features = set()

        for v in tqdm(videos):
            try:
                text_mpnet: np.ndarray = np.load(
                    Path(self.cfg.dataset.base_dir)
                    / self.cfg.dataset.name
                    / "feats"
                    / v
                    / "text_mpnet.npy"
                )
                # text_mpnet shape: (num_clips, num_subclips, 768)
                for c_idx in range(text_mpnet.shape[0]):
                    narration_embeddings.append(np.mean(text_mpnet[c_idx], axis=0))
                    narration_lookup_table.append((v, c_idx))
            except FileNotFoundError:
                videos_missing_features.add(v)

        narration_embeddings = np.array(narration_embeddings)
        return narration_embeddings, narration_lookup_table


    def cos_sim(a, b):
        return torch.mm(
            F.normalize(a, p=2, dim=1), F.normalize(b, p=2, dim=1).transpose(0, 1)
        )

    def find_step_similarities_for_segments_using_narration(
        self,
        step_des_feats,
        segment_narration_embeddings,
        segment_narration_lookup_table,
    ):
        for segment_id in tqdm(range(len(segment_narration_embeddings))):
            v, cidx = segment_narration_lookup_table[segment_id]
            save_path = Path(
                f"{self.cfg.dataset.base_dir}/{self.cfg.dataset.name}/sim_scores/{v}/segment_{cidx}.npy"
            )
            if not save_path.exists():
                cos_scores = self.cos_sim(
                    step_des_feats,
                    segment_narration_embeddings[segment_id],
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(save_path, cos_scores[:, 0].numpy())

    def get_DS_sim_scores(self):
        step_des_feats = self.get_step_des_feats(language_model="MPNet")
        segment_narration_embeddings, segment_narration_lookup_table = (
            self.gather_all_narration_MPNet_embeddings()
        )
        self.find_step_similarities_for_segments_using_narration(
            step_des_feats,
            segment_narration_embeddings,
            segment_narration_lookup_table,
        )


    def get_pseudo_label_DS_for_one_segment(self, sample_gt_path):
        step_scores = np.load(sample_gt_path)
        matched_steps, matched_steps_scores = self.find_matching_of_a_segment(
            step_scores,
            criteria=self.cfg.label_find_matched_steps_criteria,
            threshold=self.cfg.label_find_matched_steps_for_segments_threshold,
            topK=self.cfg.label_find_matched_steps_for_segments_topK,
        )

        pseudo_label_DS = {"indices": matched_steps, "values": matched_steps_scores}
        return pseudo_label_DS

    def get_pseudo_label_DS(self):
        sample_path = Path(
            f"{self.cfg.dataset.base_dir}/{self.cfg.dataset.name}/samples/samples.pickle"
        )
        if not sample_path.exists():
            self.get_samples()
        with open(sample_path, "rb") as f:
            samples = pickle.load(f)

        sim_score_path = Path(f"{self.cfg.dataset.base_dir}/{self.cfg.dataset.name}/DS_sim_scores")
        sample_pseudo_label_save_dir = Path(
            f"{self.cfg.dataset.base_dir}/{self.cfg.dataset.name}/DS_pseudo_labels"
        )
        sample_pseudo_label_save_dir.mkdir(parents=True, exist_ok=True)
        sample_pseudo_label_save_path = (
            sample_pseudo_label_save_dir
            / f"DS-{self.cfg.label_find_matched_steps_criteria}-{self.cfg.label_find_matched_steps_for_segments_threshold}-{self.cfg.label_find_matched_steps_for_segments_topK}-{self.cfg.num_nodes}.pickle"
        )

        if not sample_pseudo_label_save_path.exists():
            # start processing
            pseudo_label_DS = dict()
            for sample_index in tqdm(range(len(samples))):
                (video_sid, segment_iid) = samples[sample_index]
                segment_sid = "segment_{}".format(segment_iid)
                sample_gt_path = sim_score_path / video_sid / f"{segment_sid}.npy"

                pseudo_label_DS[sample_index] = self.get_pseudo_label_DS_for_one_segment(
                    sample_gt_path
                )

            # save
            with open(sample_pseudo_label_save_path, "wb") as f:
                pickle.dump(pseudo_label_DS, f)

        return


    def partition_dataset(self):
        """Randomly partition the dataset's features and pseudo labels."""

        with open(
            Path(self.cfg.dataset.base_dir)
            / self.cfg.dataset.name
            / "features"
            / "feats_all-mean_agg.pickle",
            "rb",
        ) as f:
            feats_all: dict[int, np.ndarray] = pickle.load(f)

        all_indices = list(feats_all.keys())
        num_each_partition = len(all_indices) // self.cfg.num_partitions
        num_allsamples = num_each_partition * self.cfg.num_partitions

        random.shuffle(all_indices)
        all_indices = all_indices[:num_allsamples]

        # VNM
        sample_pseudo_label_savepath = (
            Path(self.cfg.dataset.base_dir)
            / self.cfg.dataset.name
            / "pseudo_labels"
            / f"VNM-criteria_{self.cfg.label_find_matched_nodes_criteria}-threshold_{self.cfg.label_find_matched_nodes_for_segments_threshold}-topK_{self.cfg.label_find_matched_nodes_for_segments_topK}-size_{self.cfg.num_nodes}.pickle"
        )
        with open(sample_pseudo_label_savepath, "rb") as f:
            pseudo_labels_all_VNM = pickle.load(f)

        # VTM
        sample_pseudo_label_savepath = (
            Path(self.cfg.dataset.base_dir)
            / self.cfg.dataset.name
            / "pseudo_labels"
            / f"VTM-criteria_{self.cfg.label_find_tasks_criteria}-threshold_{self.cfg.label_find_tasks_threshold}-topK_{self.cfg.label_find_tasks_topK}.pickle"
        )
        with open(sample_pseudo_label_savepath, "rb") as f:
            pseudo_labels_all_VTM = pickle.load(f)

        # TCL
        sample_pseudo_label_savepath = (
            Path(self.cfg.dataset.base_dir)
            / self.cfg.dataset.name
            / "pseudo_labels"
            / f"TCL-criteria_{self.cfg.label_find_tasks_criteria}-threshold_{self.cfg.label_find_tasks_threshold}-topK_{self.cfg.label_find_tasks_topK}.pickle"
        )
        with open(sample_pseudo_label_savepath, "rb") as f:
            pseudo_labels_all_TCL = pickle.load(f)

        # NRL
        sample_pseudo_label_savepath = (
            Path(self.cfg.dataset.base_dir)
            / self.cfg.dataset.name
            / "pseudo_labels"
            / f"NRL-hop_{self.cfg.label_khop}-criteria_{self.cfg.label_find_neighbors_criteria}-threshold_{self.cfg.label_find_neighbors_threshold}-topK_{self.cfg.label_find_neighbors_topK}-size_{self.cfg.num_nodes}.pickle"
        )
        with open(sample_pseudo_label_savepath, "rb") as f:
            pseudo_labels_all_NRL = pickle.load(f)

        # Start partitioning
        for i in range(self.cfg.num_partitions):
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
                Path(self.cfg.dataset.base_dir)
                / self.cfg.dataset.name
                / f"index_keys_this_partition-rank_{i}-of-{self.cfg.num_partitions}.pickle",
                "wb",
            ) as f:
                pickle.dump(index_keys_this_partition, f)

            with open(
                Path(self.cfg.dataset.base_dir)
                / self.cfg.dataset.name
                / "features"
                / f"feats_all-mean_agg-rank_{i}-of-{self.cfg.num_partitions}.pickle",
                "wb",
            ) as f:
                pickle.dump(feats_this, f)

            sample_pseudo_label_savepath = (
                Path(self.cfg.dataset.base_dir)
                / self.cfg.dataset.name
                / "pseudo_labels"
                / f"VNM-criteria_{self.cfg.label_find_matched_nodes_criteria}-threshold_{self.cfg.label_find_matched_nodes_for_segments_threshold}-topK_{self.cfg.label_find_matched_nodes_for_segments_topK}-size_{self.cfg.num_nodes}-rank_{i}-of-{self.cfg.num_partitions}.pickle"
            )
            with open(sample_pseudo_label_savepath, "wb") as f:
                pickle.dump(pseudo_labels_VNM_this, f)

            sample_pseudo_label_savepath = (
                Path(self.cfg.dataset.base_dir)
                / self.cfg.dataset.name
                / "pseudo_labels"
                / f"VTM-criteria_{self.cfg.label_find_tasks_criteria}-threshold_{self.cfg.label_find_tasks_threshold}-topK_{self.cfg.label_find_tasks_topK}-rank_{i}-of-{self.cfg.num_partitions}.pickle"
            )
            with open(sample_pseudo_label_savepath, "wb") as f:
                pickle.dump(pseudo_labels_VTM_this, f)

            sample_pseudo_label_savepath = (
                Path(self.cfg.dataset.base_dir)
                / self.cfg.dataset.name
                / "pseudo_labels"
                / f"TCL-criteria_{self.cfg.label_find_tasks_criteria}-threshold_{self.cfg.label_find_tasks_threshold}-topK_{self.cfg.label_find_tasks_topK}-rank_{i}-of-{self.cfg.num_partitions}.pickle"
            )
            with open(sample_pseudo_label_savepath, "wb") as f:
                pickle.dump(pseudo_labels_TCL_this, f)

            sample_pseudo_label_savepath = (
                Path(self.cfg.dataset.base_dir)
                / self.cfg.dataset.name
                / "pseudo_labels"
                / f"NRL-hop_{self.cfg.label_khop}-criteria_{self.cfg.label_find_neighbors_criteria}-threshold_{self.cfg.label_find_neighbors_threshold}-topK_{self.cfg.label_find_neighbors_topK}-size_{self.cfg.num_nodes}-rank_{i}-of-{self.cfg.num_partitions}.pickle"
            )
            with open(sample_pseudo_label_savepath, "wb") as f:
                pickle.dump(pseudo_labels_NRL_this, f)
