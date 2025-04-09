from dataclasses import dataclass
from typing import Literal

from .base import Config


@dataclass
class ASFormerConfig(Config):
    num_decoders: int
    num_layers: int
    r1: int
    r2: int
    num_f_maps: int
    channel_masking_rate: float
    att_type: str
    alpha: float
    p: float
    scheduler_mode: Literal["min", "max"]
    scheduler_factor: float
    scheduler_patience: int
    mse_weight: float
