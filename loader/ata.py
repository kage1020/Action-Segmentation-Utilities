from torch.utils.data import DataLoader

from configs import Config

from .main import BaseDataLoader, BaseDataset


class AnomalousToyAssemblyDataset(BaseDataset):
    def __init__(self, cfg: Config, train: bool = True):
        super().__init__(cfg, "AnomalousToyAssemblyDataset", train)


class AnomalousToyAssemblyDataLoader(BaseDataLoader, DataLoader):
    dataset: AnomalousToyAssemblyDataset

    def __init__(self, cfg: Config, train: bool = True, collate_fn=None):
        dataset = AnomalousToyAssemblyDataset(cfg, train)
        super(BaseDataLoader, self).__init__(
            dataset=dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=cfg.dataset.shuffle,
            collate_fn=collate_fn,
        )
