from dataclasses import dataclass

from base.main import Config


@dataclass
class LTCConfig:
    num_stages: int
    num_layers: int
    model_dim: int
    dim_reduction: int
    dropout_prob: float
    conv_dilation_factor: int
    windowed_attn_w: int
    long_term_attn_g: int
    use_instance_norm: bool
    channel_masking_prob: float


@dataclass
class AttentionConfig:
    num_attn_heads: int
    dropout: float


@dataclass
class SolverConfig:
    t_max: int
    eta_min: int
    milestone: int


@dataclass
class LTContextConfig(Config):
    mse_clip_val: float
    mse_weight: float
    LTC: LTCConfig
    ATTENTION: AttentionConfig
    SOLVER: SolverConfig
