import pickle
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from trainer import PaprikaConfig
from .helper import get_step_des_feats


def get_nodes_by_removing_step_duplicates(cfg: PaprikaConfig, step_des_feats=None):
    if cfg.remove_step_duplicates:
        node2step_path = Path(cfg.document_dir) / "node2step.pickle"
        step2node_path = Path(cfg.document_dir) / "step2node.pickle"

        if node2step_path.exists() and step2node_path.exists():
            with open(node2step_path, "rb") as f:
                node2step = pickle.load(f)
            with open(step2node_path, "rb") as f:
                step2node = pickle.load(f)
        else:
            assert step_des_feats is not None

            clustering = AgglomerativeClustering(
                n_clusters=None,
                linkage=cfg.step_clustering_linkage,
                distance_threshold=cfg.step_clustering_distance_thresh,
                affinity=cfg.step_clustering_affinity,
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
        node2step = {i: [i] for i in range(cfg.num_nodes)}
        step2node = {i: i for i in range(cfg.num_nodes)}

    return node2step, step2node


def get_nodes(cfg: PaprikaConfig):
    step_des_feats = get_step_des_feats(cfg, language_model="MPNet")
    node2step, step2node = get_nodes_by_removing_step_duplicates(cfg, step_des_feats)
    return node2step, step2node
