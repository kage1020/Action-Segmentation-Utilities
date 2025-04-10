from torch.utils.data import DataLoader

from ..configs import Config
from .main import BaseDataLoader, BaseDataset


class Howto100MDataset(BaseDataset):
    def __init__(self, cfg: Config, train: bool = True):
        super().__init__(cfg, "Howto100MDataset", train)


class Howto100MDataLoader(BaseDataLoader, DataLoader):
    dataset: Howto100MDataset

    def __init__(self, cfg: Config, train: bool = True, collate_fn=None):
        dataset = Howto100MDataset(cfg, train)
        super(BaseDataLoader, self).__init__(
            dataset=dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=cfg.dataset.shuffle,
            collate_fn=collate_fn,
        )
