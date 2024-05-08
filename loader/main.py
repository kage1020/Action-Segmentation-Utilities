import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DatasetConfig:
    dataset: str
    base_dir: str = "data"
    backgrounds: list[str] = []
    split: int = 1
    split_dir: str = "splits"
    gt_dir: str = "groundTruth"
    feature_dir: str = "features"
    pseudo_dir: str = "pseudo"
    prob_dir: str = "prob"
    action: str = "train"
    sampling_rate: int = 1
    batch_size: int = 1
    shuffle: bool = False
    device: str
    refine_penalty: float = 0.1

    def __init__(
        self,
        action: str,
        dataset: str,
        split: str,
        device: str,
        base_dir: str = "data",
        backgrounds: list[str] = [],
        gt_dir: str = "groundTruth",
        feature_dir: str = "features",
        split_dir: str = "splits",
        pseudo_dir: str = "pseudo",
        prob_dir: str = "prob",
        sampling_rate: int = 1,
        batch_size: int = 1,
        refine_penalty: float = 0.1
    ):
        self.action = action
        self.dataset = dataset
        self.base_dir = base_dir
        self.backgrounds = backgrounds
        self.split = split
        self.gt_dir = gt_dir
        self.feature_dir = feature_dir
        self.split_dir = split_dir
        self.pseudo_dir = pseudo_dir
        self.prob_dir = prob_dir
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.device = device
        self.shuffle = self.action == "train"
        refine_penalty = refine_penalty


def get_mapping(file_path: str, has_header: bool = False):
    '''
    Get the mapping of classes to integers

    Mapping file format should be:
    class1 0
    class2 1
    ...
    '''
    with open(file_path, "r") as f:
        lines = f.readlines()
        if has_header:
            lines = lines[1:]
    text_to_int = {}
    int_to_text = {}
    for line in lines:
        text, num = line.strip().split()
        text_to_int[text] = int(num)
        int_to_text[int(num)] = text
    return text_to_int, int_to_text


def get_actions(file_path: str, has_header: bool = False):
    '''
    Get the actions per video

    Actions file format should be:
    action1 class1,class2,class3,...
    action2 class4,class5,class6,...
    '''
    with open(file_path, "r") as f:
        lines = f.readlines()
        if has_header:
            lines = lines[1:]
    actions = {}
    for line in lines:
        action, classes = line.strip().split()
        actions[action] = list(map(int, classes.split(",")))
    return actions

def get_matching(file_path: str, has_header: bool = False):
    '''
    Get the matching of classes

    Matching file format should be:
    video1 action1
    video2 action2
    ...
    '''
    with open(file_path, "r") as f:
        lines = f.readlines()
        if has_header:
            lines = lines[1:]
    matching = {}
    for line in lines:
        video, action = line.strip().split()
        matching[video] = action


def mask_backgrounds_in_mapping(mapping: dict, backgrounds: list[int | str], mask_value: int | str):
    '''
    Mask the background classes in the mapping. Key of mapping should be same as the elements of backgrounds
    '''
    _mapping = mapping.copy()
    for bg in backgrounds:
        _mapping[bg] = mask_value
    return _mapping


def remove_backgrounds_in_actions(actions: dict, backgrounds: list[int | str]):
    '''
    Remove the background classes from the actions. Key of actions should be same as the elements of backgrounds
    '''
    _actions = actions.copy()
    for bg in backgrounds:
        for i, classes in enumerate(_actions.values()):
            if bg in classes:
                _actions[i] = [c for c in classes if c != bg]
    return _actions


class BaseDataset(Dataset):
    cfg: DatasetConfig = None
    class_to_int = None
    int_to_class = None
    masked_class_to_int = None
    masked_int_to_class = None
    actions = {}
    masked_actions = {}
    videos = []

    def __init__(self):
        # load videos
        with open(f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.split_dir}/{self.cfg.action}.split{self.cfg.split}.bundle", "r") as f:
            lines = f.readlines()
            self.videos = [line.strip() for line in lines if line.strip() != ""]

        if self.cfg.action == "train":
            self.__generate_pseudo_labels()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # load features
        video = Path(self.videos[idx])
        video_name = video.stem
        features = np.load(f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.feature_dir}/{video_name}.npy")
        features = torch.from_numpy(features).float()
        features = features[::self.cfg.sampling_rate].unsqueeze(0)

        # load ground truth/pseudo labels
        gt_path = Path(f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.gt_dir}/{video.stem}.txt")
        pseudo_path = Path(f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.pseudo_dir}/{video.stem}.txt")
        prob_path = Path(f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.prob_dir}/{video.stem}.npy")
        if gt_path.exists():
            with open(gt_path, "r") as f:
                labels = [line.strip() for line in f.readlines() if line.strip() != ""]
            probabilities = torch.zeros(len(labels), len(self.class_to_int))
        else:
            with open(pseudo_path, "r") as f:
                labels = [line.strip() for line in f.readlines() if line.strip() != ""]
            probabilities = torch.from_numpy(np.load(prob_path)).float()
        labels = [self.class_to_int[label] for label in labels]
        labels = torch.tensor(labels).long()[::self.cfg.sampling_rate].unsqueeze(0)

        return features, labels, probabilities

    def __generate_pseudo_labels(self, model):
        os.makedirs(f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.pseudo_dir}", exist_ok=True)
        os.makedirs(f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.prob_dir}", exist_ok=True)

        model.eval()
        with torch.no_grad():
            for video_path in self.videos:
                video = Path(video_path)
                gt_path = Path(f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.gt_dir}/{video.stem}.txt")
                if gt_path.exists():
                    continue

                # load features
                features = np.load(f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.feature_dir}/{video.stem}.npy")
                features = torch.from_numpy(features).float()
                features = features[::self.cfg.sampling_rate].unsqueeze(0)
                features = features.to(self.cfg.device)

                # create pseudo labels
                pseudo_labels = model(features)
                pseudo_probs = F.softmax(pseudo_labels, dim=1)

                matching = get_matching(f"{self.cfg.base_dir}/{self.cfg.dataset}/matching.txt")
                action = matching[video.stem]
                pseudo_probs = self.__refine_pseudo_labels(pseudo_probs, action)
                pseudo_labels = pseudo_probs.argmax(dim=1).unsqueeze(0)

                with open(gt_path, "w") as f:
                    for label in pseudo_labels:
                        f.write(f"{self.masked_int_to_class[label.item()]}\n")
                np.save(f"{self.cfg.base_dir}/{self.cfg.dataset}/{self.cfg.prob_dir}/{video.stem}.npy", pseudo_probs.cpu().numpy())

    def __refine_pseudo_labels(self, probs, action):
        _probs = probs.clone()
        for i in range(len(probs)):
            for j in range(len(probs[i])):
                if j not in self.masked_actions[action] and j != self.masked_class_to_int["background"]:
                    _probs[i, j] *= self.cfg.refine_penalty
            if _probs.sum() < 1e-5:
                _probs[i, self.masked_class_to_int["background"]] += 1e-5
            _probs[i] /= _probs[i].sum()
        return _probs


class SaladsDataset(BaseDataset):
    def __init__(self, cfg: DatasetConfig = DatasetConfig()):
        self.cfg = cfg
        self.class_to_int, self.int_to_class = get_mapping(f"{self.cfg.base_dir}/{self.cfg.dataset}/mapping.txt")
        self.masked_class_to_int = mask_backgrounds_in_mapping(self.class_to_int, cfg.backgrounds, -1)
        self.masked_int_to_class = mask_backgrounds_in_mapping(self.int_to_class, cfg.backgrounds, "background")
        super().__init__()


class SaladsDataLoader(DataLoader):
    def __init__(self, cfg: DatasetConfig = DatasetConfig()):
        dataset = SaladsDataset(cfg)
        super(SaladsDataLoader, self).__init__(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
        self.cfg = cfg
        self.class_to_int = dataset.class_to_int
        self.int_to_class = dataset.int_to_class
        self.masked_class_to_int = dataset.masked_class_to_int
        self.masked_int_to_class = dataset.masked_int_to_class


class GTEADataset(BaseDataset):
    def __init__(self, cfg: DatasetConfig = DatasetConfig(backgrounds=["background"])):
        self.cfg = cfg
        self.class_to_int, self.int_to_class = get_mapping(f"{self.cfg.base_dir}/{self.cfg.dataset}/mapping.txt")
        self.masked_class_to_int = mask_backgrounds_in_mapping(self.class_to_int, cfg.backgrounds, -1)
        self.masked_int_to_class = mask_backgrounds_in_mapping(self.int_to_class, cfg.backgrounds, "background")
        super().__init__()


class GTEADataLoader(DataLoader):
    def __init__(self, cfg: DatasetConfig = DatasetConfig(backgrounds=["background"])):
        dataset = GTEADataset(cfg)
        super(GTEADataLoader, self).__init__(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
        self.cfg = cfg
        self.class_to_int = dataset.class_to_int
        self.int_to_class = dataset.int_to_class
        self.masked_class_to_int = dataset.masked_class_to_int
        self.masked_int_to_class = dataset.masked_int_to_class


class BreakfastDataset(BaseDataset):
    def __init__(self, cfg: DatasetConfig = DatasetConfig(backgrounds=["SIL"])):
        self.cfg = cfg
        self.class_to_int, self.int_to_class = get_mapping(f"{self.cfg.base_dir}/{self.cfg.dataset}/mapping.txt")
        self.masked_class_to_int = mask_backgrounds_in_mapping(self.class_to_int, cfg.backgrounds, -1)
        self.masked_int_to_class = mask_backgrounds_in_mapping(self.int_to_class, cfg.backgrounds, "SIL")
        self.actions = get_actions(f"{self.cfg.base_dir}/{self.cfg.dataset}/actions.txt")
        self.masked_actions = remove_backgrounds_in_actions(self.actions, cfg.backgrounds)
        super().__init__()


class BreakfastDataLoader(DataLoader):
    def __init__(self, cfg: DatasetConfig = DatasetConfig(backgrounds=["SIL"])):
        dataset = BreakfastDataset(cfg)
        super(BreakfastDataLoader, self).__init__(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
        self.cfg = cfg
        self.class_to_int = dataset.class_to_int
        self.int_to_class = dataset.int_to_class
        self.masked_class_to_int = dataset.masked_class_to_int
        self.masked_int_to_class = dataset.masked_int_to_class


class Assembly101Dataset(BaseDataset):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Assembly101Dataset is not implemented yet.")


class AnomalousToyAssemblyDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("AnomalousToyAssemblyDataset is not implemented yet.")


class NissanDataset(BaseDataset):
    def __init__(self, cfg: DatasetConfig = DatasetConfig(backgrounds=["background"])):
        self.cfg = cfg
        self.class_to_int, self.int_to_class = get_mapping(f"{self.cfg.base_dir}/{self.cfg.dataset}/mapping.txt")
        self.masked_class_to_int = mask_backgrounds_in_mapping(self.class_to_int, cfg.backgrounds, -1)
        self.masked_int_to_class = mask_backgrounds_in_mapping(self.int_to_class, cfg.backgrounds, "background")
        self.actions = get_actions(f"{self.cfg.base_dir}/{self.cfg.dataset}/actions.txt")
        self.masked_actions = remove_backgrounds_in_actions(self.actions, cfg.backgrounds)
        super().__init__()


class NissanDataLoader(DataLoader):
    def __init__(self, cfg: DatasetConfig = DatasetConfig(backgrounds=["background"])):
        dataset = NissanDataset(cfg)
        super(NissanDataLoader, self).__init__(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
        self.cfg = cfg
        self.class_to_int = dataset.class_to_int
        self.int_to_class = dataset.int_to_class
        self.masked_class_to_int = dataset.masked_class_to_int
        self.masked_int_to_class = dataset.masked_int_to_class
