import torch.nn as nn
from torch import Tensor

from configs.paprika import PaprikaConfig


class Adapter(nn.Module):
    def __init__(self, cfg: PaprikaConfig):
        super().__init__()
        self.cfg = cfg

        self.adapter = nn.Sequential(
            nn.Linear(cfg.dataset.input_dim, cfg.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.bottleneck_dim, cfg.dataset.input_dim),
        )

        # Procedural Knowledge Graph
        if "PKG" in cfg.adapter_objective:
            # Video Node Matching
            if "VNM" in cfg.adapter_objective:
                output_dim = cfg.dataset.num_nodes
                self.answer_head_VNM = nn.Sequential(
                    nn.Linear(cfg.dataset.input_dim, output_dim // 4),
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
                                cfg.dataset.input_dim,
                                cfg.document.num_tasks // 2,
                            ),
                            nn.ReLU(inplace=True),
                            nn.Linear(
                                cfg.document.num_tasks // 2, cfg.document.num_tasks
                            ),
                        ),
                        nn.Sequential(
                            nn.Linear(
                                cfg.dataset.input_dim, cfg.dataset.num_tasks // 2
                            ),
                            nn.ReLU(inplace=True),
                            nn.Linear(
                                cfg.dataset.num_tasks // 2, cfg.dataset.num_tasks
                            ),
                        ),
                    ]
                )

            # Task Context Learning
            if "TCL" in cfg.adapter_objective:
                output_dim = cfg.dataset.num_nodes
                self.answer_head_TCL = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(cfg.dataset.input_dim, output_dim // 4),
                            nn.ReLU(inplace=True),
                            nn.Linear(output_dim // 4, output_dim // 2),
                            nn.ReLU(inplace=True),
                            nn.Linear(output_dim // 2, output_dim),
                        ),
                        nn.Sequential(
                            nn.Linear(cfg.dataset.input_dim, output_dim // 4),
                            nn.ReLU(inplace=True),
                            nn.Linear(output_dim // 4, output_dim // 2),
                            nn.ReLU(inplace=True),
                            nn.Linear(output_dim // 2, output_dim),
                        ),
                    ]
                )

            # Node Relation Learning
            if "NRL" in cfg.adapter_objective:
                output_dim = cfg.dataset.num_nodes
                self.answer_head_NRL = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(cfg.dataset.input_dim, output_dim // 4),
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
            output_dim = cfg.dataset.num_classes
            self.answer_head = nn.Sequential(
                nn.Linear(cfg.dataset.input_dim, output_dim // 6),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim // 6, output_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim // 4, output_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim // 2, output_dim),
            )

    def forward(
        self, features: Tensor, prediction: bool = True
    ) -> Tensor | tuple[Tensor, list[Tensor], list[Tensor], list[Tensor]]:
        """
        features: (B, 512)
        """
        refined_features = self.adapter(features)

        if not prediction:
            return refined_features

        if "PKG" in self.cfg.adapter_objective:
            VNM_answer = nn.Identity()
            VTM_answer = [nn.Identity() for _ in range(2)]
            TCL_answer = [nn.Identity() for _ in range(2)]
            NRL_answer = [nn.Identity() for _ in range(2 * self.cfg.pretrain_khop)]
            if "VNM" in self.cfg.adapter_objective:
                VNM_answer = self.answer_head_VNM(refined_features)
            if "VTM" in self.cfg.adapter_objective:
                VTM_answer = [head(refined_features) for head in self.answer_head_VTM]
            if "TCL" in self.cfg.adapter_objective:
                TCL_answer = [head(refined_features) for head in self.answer_head_TCL]
            if "NRL" in self.cfg.adapter_objective:
                NRL_answer = [head(refined_features) for head in self.answer_head_NRL]
            return VNM_answer, VTM_answer, TCL_answer, NRL_answer
        else:
            return self.answer_head(refined_features)
