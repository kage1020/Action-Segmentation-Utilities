# Action Segmentation Utilities

Utilities for action segmentation task. This repository contains the following utilities:

- [Evaluation Metrics](evaluator/README.md)
- [Video Feature Extractor](extractor/README.md)
- [Dataset Loaders](loader/README.md)
- [Loggers](logger/README.md)
- [Running Managers](manager/README.md)
- [Models](models/README.md)
- [Annotation Processor](processor/README.md)
- [Trainers](trainer/README.md)
- [Visualizers](visualizer/README.md)

## How to Use

### Use on Docker with Dev Containers

This repository is configured to run on Docker with Dev Containers. To use this repository on Docker, follow the steps below:

1. Install [Docker](https://www.docker.com/products/docker-desktop/) on your machine.
2. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension on Visual Studio Code.
3. Open this repository on Visual Studio Code as a Dev Container.
4. Run the following command in the terminal to install the required packages:

    ```bash
    poetry shell
    poetry install
    ```

That's it! You can now use this repository on Docker with Dev Containers.

### Use as a module (Using Git Submodule)

You can use this repository as a submodule in your project. To add this repository as a submodule, run the following command:

```bash
# In the root directory of your project
git submodule add https://github.com/kage1020/Action-Segmentation-Utilities.git asu
cd asu

# Install the required packages
poetry install
# or
pip install -r requirements.txt
```

That's it! You can now use this repository as a submodule in your project.

### Run your code with Hydra

You can use this repository with [Hydra](https://hydra.cc/) to manage your configurations. To use Hydra, follow the steps below:

```python
# main.py
import hydra
from base import Base, Config
from loader import BreakfastDataLoader
from models import ASFormer
from trainer import ASFormerTrainer


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: Config) -> None:
    train_loader = BreakfastDataLoader(cfg)
    test_loader = BreakfastDataLoader(cfg, train=False)
    device = Base.get_device(cfg.device)
    model = ASFormer(
        num_decoders=cfg.num_decoders,
        num_layers=cfg.num_layers,
        r1=cfg.r1,
        r2=cfg.r2,
        num_f_maps=cfg.num_f_maps,
        channel_masking_rate=cfg.channel_masking_rate,
        att_type=cfg.att_type,
        alpha=cfg.alpha,
        p=cfg.p,
        num_classes=cfg.dataset.num_classes,
        input_dim=cfg.dataset.input_dim,
    )
    model = Base.load_model(model, cfg.best_model_path, device, Base.get_logger("main"))
    trainer = ASFormerTrainer(cfg, model)

    trainer.train(train_loader, test_loader)


if __name__ == "__main__":
    main()
```

If you use this repository as a submodule, add `asu` to the module path like below:

```python
from asu.base import Base, Config
from asu.loader import BreakfastDataLoader
from asu.models import ASFormer
from asu.trainer import ASFormerTrainer
```

Run the following command to execute the code:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config-path=config --config-name=default
```

If you override the default configuration, use [Basic Override syntax](https://hydra.cc/docs/1.3/advanced/override_grammar/basic/).
