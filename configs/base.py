from dataclasses import asdict, astuple, dataclass
from omegaconf import DictConfig


@dataclass
class DatasetConfig(DictConfig):
    name: str
    mapping_path: str | None
    actions_path: str | None
    matching_path: str | None
    has_mapping_header: bool | None
    mapping_separator: str | None
    has_actions_header: bool | None
    actions_action_separator: str | None
    actions_class_separator: str | None
    matching_separator: str | None
    num_classes: int
    input_dim: int
    split: int
    num_fold: int
    backgrounds: list[str]
    batch_size: int
    sampling_rate: int
    shuffle: bool
    base_dir: str
    split_dir: str
    gt_dir: str
    feature_dir: str
    action_dir: str
    prob_dir: str | None
    pseudo_dir: str | None
    semi_per: float | None


@dataclass
class Config(DictConfig):
    # system
    seed: int
    device: str
    verbose: bool
    val_skip: bool

    dataset: DatasetConfig

    # learning
    train: bool
    model_name: str
    model_path: str
    result_dir: str
    epochs: int
    lr: float
    weight_decay: float
    warmup_steps: int | None
    use_pseudo: bool
    refine_pseudo: bool

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __post_init__(self):
        self.dataset = DatasetConfig(**asdict(self.dataset))

    def __iter__(self):
        return iter(astuple(self))

    def __or__(self, other):
        return self.__class__(**asdict(self) | asdict(other))
