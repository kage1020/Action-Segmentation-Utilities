from dataclasses import dataclass
from omegaconf import DictConfig

from configs.base import Config, DatasetConfig


@dataclass
class DatasetPaprikaConfig(DatasetConfig):
    num_nodes: int
    num_tasks: int


@dataclass
class DocumentConfig(DictConfig):
    name: str
    num_tasks: int


@dataclass
class PaprikaConfig(Config):
    dataset: DatasetPaprikaConfig  # number of classes in video dataset
    document: DocumentConfig
    adapter_objective: str

    # distributed training parameters
    partition_dataset: bool
    num_partitions: int

    # model parameters
    bottleneck_dim: int
    pretrain_khop: int


@dataclass
class PaprikaPseudoConfig(PaprikaConfig):
    remove_step_duplicates: bool
    video_meta_csv_path: str
    task_id_to_task_name_csv_path: str

    # step clustering parameters
    step_clustering_linkage: str
    step_clustering_distance_threshold: float
    step_clustering_affinity: str
    edge_min_aggconf: float

    # segment matching parameters
    graph_find_matched_steps_criteria: str
    graph_find_matched_steps_for_segments_threshold: float
    graph_find_matched_steps_for_segments_topK: int

    # VNM parameters
    label_find_matched_nodes_criteria: str
    label_find_matched_nodes_for_segments_threshold: float
    label_find_matched_nodes_for_segments_topK: int

    # VTM parameters
    label_find_tasks_criteria: str
    label_find_tasks_threshold: float
    label_find_tasks_topK: int

    # TCL parameters
    label_find_tasknodes_criteria: str
    label_find_tasknodes_threshold: float
    label_find_tasknodes_topK: int
    label_num_dataset_tasks_to_consider: int

    # NRL parameters
    label_khop: int
    label_find_neighbors_criteria: str
    label_find_neighbors_threshold: float
    label_find_neighbors_topK: int


@dataclass
class PaprikaDownstreamConfig(PaprikaConfig):
    label_find_matched_steps_criteria: str
    label_find_matched_steps_for_segments_threshold: float
    label_find_matched_steps_for_segments_topK: int
