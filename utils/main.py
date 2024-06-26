import glob
import random
import numpy as np
import torch
from torch.nn import Module


def init_seed(seed: int = 42):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def load_model(model: Module, model_path: str, device="cpu"):
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def load_best_model(model: Module, model_dir: str, device="cpu"):
    model_paths = glob.glob(f"{model_dir}/*")
    model_paths.sort()
    best_model = next(filter(lambda x: "best" in x, model_paths), None)
    if best_model:
        model = load_model(model, best_model, device)
    else:
        print("Best model was not found")

    return model


def save_model(model: Module, model_path: str):
    model = model.cpu()
    torch.save(model.state_dict(), model_path)
