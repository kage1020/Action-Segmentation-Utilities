from torch.utils.data import DataLoader

from base.main import Config
from .main import BaseDataset, BaseDataLoader


class SaladsDataset(BaseDataset):
    def __init__(self, cfg: Config, train: bool = True):
        super().__init__(cfg, "SaladsDataset", train)


class SaladsDataLoader(BaseDataLoader, DataLoader):
    dataset: SaladsDataset

    def __init__(self, cfg: Config, train: bool = True, collate_fn=None):
        dataset = SaladsDataset(cfg, train)
        super(BaseDataLoader, self).__init__(
            dataset=dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=cfg.dataset.shuffle,
            collate_fn=collate_fn,
        )
