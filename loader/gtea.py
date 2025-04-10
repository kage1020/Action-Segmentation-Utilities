from torch.utils.data import DataLoader

from ..configs import Config
from .main import BaseDataLoader, BaseDataset


class GteaDataset(BaseDataset):
    def __init__(self, cfg: Config, train: bool = True):
        super().__init__(cfg, "GteaDataset", train)


class GteaDataLoader(BaseDataLoader, DataLoader):
    dataset: GteaDataset

    def __init__(self, cfg: Config, train: bool = True, collate_fn=None):
        dataset = GteaDataset(cfg, train)
        super(BaseDataLoader, self).__init__(
            dataset=dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=cfg.dataset.shuffle,
            collate_fn=collate_fn,
        )
