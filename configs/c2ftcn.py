from dataclasses import dataclass
from base.main import Config

@dataclass
class C2FTCNConfig(Config):
    feature_size: int
    output_size: int
    num_iter: int
    start_iter: int
    unsupervised_skip: bool
    epochs_unsupervised: int
    semi_per: float
    lr_proj: float
    lr_main: float
    lr_unsupervised: float
    gamma_proj: float
    gamma_main: float
    steps_proj: list[int]
    steps_main: list[int]
    epsilon: float
    epsilon_l: float
    delta: float
    weights: list[float]
    high_level_act_loss: bool
    num_samples_frames: int
