# from .DS_get_sim_scores import DS_get_sim_scores
from trainer import PaprikaPretrainConfig
from .get_sim_scores import get_DS_sim_scores, get_sim_scores
from .get_nodes import get_nodes
from .get_edges import get_edges

# from .pseudo_label_DS import get_pseudo_label_DS
from .pseudo_label_VNM import get_pseudo_label_VNM
from .pseudo_label_VTM import get_pseudo_label_VTM
from .pseudo_label_TCL import get_pseudo_label_TCL
from .pseudo_label_NRL import get_pseudo_label_NRL
from .pseudo_label_DS import get_pseudo_label_DS
from .dataset_utils import partition_dataset


def obtain_external_knowledge(cfg: PaprikaPretrainConfig):
    if  cfg.segment_step_sim_scores_DS_ready:
        get_DS_sim_scores(cfg)

    if cfg.segment_step_sim_scores_ready:
        get_sim_scores(cfg)

    if cfg.nodes_formed:
        get_nodes(cfg)

    if cfg.edges_formed:
        get_edges(cfg)

    if cfg.pseudo_label_DS_ready:
        get_pseudo_label_DS(cfg)

    if cfg.pseudo_label_VNM_ready:
        get_pseudo_label_VNM(cfg)

    if cfg.pseudo_label_VTM_ready:
        get_pseudo_label_VTM(cfg)

    if cfg.pseudo_label_TCL_ready:
        get_pseudo_label_TCL(cfg)

    if cfg.pseudo_label_NRL_ready:
        get_pseudo_label_NRL(cfg)

    if cfg.partition_dataset:
        partition_dataset(cfg)
