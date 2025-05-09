from torch.utils.data import DataLoader

from ..configs import Config
from .main import BaseDataLoader, BaseDataset


class BreakfastDataset(BaseDataset):
    def __init__(self, cfg: Config, train: bool = True):
        super().__init__(cfg, "BreakfastDataset", train)


class BreakfastDataLoader(BaseDataLoader, DataLoader):
    dataset: BreakfastDataset

    def __init__(self, cfg: Config, train: bool = True, collate_fn=None):
        dataset = BreakfastDataset(cfg, train)
        super(BaseDataLoader, self).__init__(
            dataset=dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=cfg.dataset.shuffle if train else False,
            collate_fn=collate_fn,
        )
