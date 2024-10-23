from torch.nn import Module
from dataclasses import dataclass
from base import Config

# from models.paprika.build_knowledge import obtain_external_knowledge
from trainer import Trainer


@dataclass
class PaprikaConfig(Config):
    document_dir: str
    document_num_tasks: int
    dataset_num_tasks: int

    num_nodes: int


@dataclass
class PaprikaPseudoConfig(PaprikaConfig):
    # remove_step_duplicates: bool
    # step_clustering_linkage: str
    # step_clustering_distance_thresh: float
    # step_clustering_affinity: str
    # edge_min_aggconf: int
    pass


@dataclass
class PaprikaPretrainConfig(PaprikaConfig):
    # label_find_matched_nodes_criteria: str
    # label_find_matched_nodes_for_segments_thresh: float
    # label_find_matched_nodes_for_segments_topK: int
    # label_find_tasks_criteria: str
    # label_find_tasks_thresh: float
    # label_find_tasks_topK: int
    # label_find_tasknodes_criteria: str
    # label_find_tasknodes_thresh: float
    # label_find_tasknodes_topK: int
    # label_find_neighbors_criteria: str
    # label_find_neighbors_thresh: float
    # label_find_neighbors_topK: int
    # label_khop: int

    adapter_objective: str
    pretrain_khop: int

    s3d_hidden_dim: int
    bottleneck_dim: int
    adapter_refined_feat_dim: int
    adapter_num_classes: int | None
    # video_meta_csv_path: str
    # task_id_to_task_name_csv_path: str
    # num_partitions: int
    segment_step_sim_scores_ready: bool
    segment_step_sim_scores_DS_ready: bool
    nodes_formed: bool
    edges_formed: bool
    pseudo_label_VNM_ready: bool
    pseudo_label_VTM_ready: bool
    pseudo_label_TCL_ready: bool
    pseudo_label_NRL_ready: bool
    pseudo_label_DS_ready: bool
    partition_dataset: bool


class PaprikaTrainer(Trainer):
    def __init__(self, cfg: PaprikaConfig, model: Module):
        super().__init__(cfg, model, name="PaprikaTrainer")
        self.cfg = cfg

    def make_pseudo_label(self):
        pass
        # obtain_external_knowledge(self.cfg)

    def train(self):
        pass
