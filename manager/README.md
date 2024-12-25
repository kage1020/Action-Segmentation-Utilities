# Experiment manager for ASU

This module provides the experiment management tools for action segmentation.

## Hydra

> [!WARNING]
> This module is under development.

You can use this module with [Hydra](https://hydra.cc/) to manage your experiments using configuration files. To use this module, you need to add decorator to your function.

```python
from manager.hydra import HydraManager
from configs.base import Config


@HydraManager(config_path="config.yaml")
def main(cfg: Config):
    ...

if __name__ == "__main__":
    main()
```

## MLflow

> [!WARNING]
> This module is under development.

You can use this module with [MLflow](https://mlflow.org/) to manage your model parameters and metrics.

```python
from manager.mlflow import MLflowManager

mlflow = MLflowManager()
mlflow.set(tracking_url="http://localhost:5000", experiment_name="default")

with mlflow.start_run():
    mlflow.log_params({"lr": 0.01, "batch_size": 32})
    mlflow.log_metrics({"loss": 0.1, "accuracy": 0.9})
```
