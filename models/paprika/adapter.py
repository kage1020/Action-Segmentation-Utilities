import torch.nn as nn

from trainer import PaprikaPretrainConfig

from torch import Tensor


class Adapter(nn.Module):
    def __init__(self, cfg: PaprikaPretrainConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.adapter_refined_feat_dim == cfg.s3d_hidden_dim

        self.adapter = nn.Sequential(
            nn.Linear(cfg.s3d_hidden_dim, cfg.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.bottleneck_dim, cfg.adapter_refined_feat_dim),
        )

        # Procedural Knowledge Graph
        if "PKG" in cfg.adapter_objective:
            # Video Node Matching
            if "VNM" in cfg.adapter_objective:
                output_dim = cfg.num_nodes
                self.answer_head_VNM = nn.Sequential(
                    nn.Linear(cfg.adapter_refined_feat_dim, output_dim // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(output_dim // 4, output_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(output_dim // 2, output_dim),
                )

            # Video Task Matching
            if "VTM" in cfg.adapter_objective:
                self.answer_head_VTM = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(
                                cfg.adapter_refined_feat_dim,
                                cfg.document_num_tasks // 2,
                            ),
                            nn.ReLU(inplace=True),
                            nn.Linear(
                                cfg.document_num_tasks // 2, cfg.document_num_tasks
                            ),
                        ),
                        nn.Sequential(
                            nn.Linear(
                                cfg.adapter_refined_feat_dim, cfg.dataset_num_tasks // 2
                            ),
                            nn.ReLU(inplace=True),
                            nn.Linear(
                                cfg.dataset_num_tasks // 2, cfg.dataset_num_tasks
                            ),
                        ),
                    ]
                )

            # Task Context Learning
            if "TCL" in cfg.adapter_objective:
                output_dim = cfg.num_nodes
                self.answer_head_TCL = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(cfg.adapter_refined_feat_dim, output_dim // 4),
                            nn.ReLU(inplace=True),
                            nn.Linear(output_dim // 4, output_dim // 2),
                            nn.ReLU(inplace=True),
                            nn.Linear(output_dim // 2, output_dim),
                        ),
                        nn.Sequential(
                            nn.Linear(cfg.adapter_refined_feat_dim, output_dim // 4),
                            nn.ReLU(inplace=True),
                            nn.Linear(output_dim // 4, output_dim // 2),
                            nn.ReLU(inplace=True),
                            nn.Linear(output_dim // 2, output_dim),
                        ),
                    ]
                )

            # Node Relation Learning
            if "NRL" in cfg.adapter_objective:
                output_dim = cfg.num_nodes
                self.answer_head_NRL = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(cfg.adapter_refined_feat_dim, output_dim // 4),
                            nn.ReLU(inplace=True),
                            nn.Linear(output_dim // 4, output_dim // 2),
                            nn.ReLU(inplace=True),
                            nn.Linear(output_dim // 2, output_dim),
                        )
                        for _ in range(2 * cfg.pretrain_khop)
                    ]
                )

        # Downstream Task
        else:
            assert cfg.adapter_num_classes is not None
            self.answer_head = nn.Sequential(
                nn.Linear(cfg.adapter_refined_feat_dim, cfg.adapter_num_classes // 6),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.adapter_num_classes // 6, cfg.adapter_num_classes // 4),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.adapter_num_classes // 4, cfg.adapter_num_classes // 2),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.adapter_num_classes // 2, cfg.adapter_num_classes),
            )

    def forward(self, features: Tensor, prediction: bool = False):
        """
        features: (B, 512)
        """
        refined_features = self.adapter(features)

        if not prediction:
            return refined_features

        if "PKG" in self.cfg.adapter_objective:
            if self.cfg.adapter_objective == "PKG_VNM_VTM_TCL_NRL":
                VNM_answer = self.answer_head_VNM(refined_features)
                VTM_answer = [head(refined_features) for head in self.answer_head_VTM]
                TCL_answer = [head(refined_features) for head in self.answer_head_TCL]
                NRL_answer = [head(refined_features) for head in self.answer_head_NRL]
                return VNM_answer, VTM_answer, TCL_answer, NRL_answer
            else:
                return self.answer_head(refined_features)
