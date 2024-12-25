from torch.utils.data import DataLoader

from configs.base import Config
from .main import BaseDataset, BaseDataLoader


class Assembly101Dataset(BaseDataset):
    def __init__(self, cfg: Config, train: bool = True):
        super().__init__(cfg, "Assembly101Dataset", train)


class Assembly101DataLoader(BaseDataLoader, DataLoader):
    dataset: Assembly101Dataset

    def __init__(self, cfg: Config, train: bool = True, collate_fn=None):
        dataset = Assembly101Dataset(cfg, train)
        super(BaseDataLoader, self).__init__(
            dataset=dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=cfg.dataset.shuffle,
            collate_fn=collate_fn,
        )
