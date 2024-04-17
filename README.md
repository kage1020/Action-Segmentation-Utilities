# Action Segmentation Utilities

Utilities for action segmentation tasks. This repository contains the following utilities:

- [Dataset Loader](loader/README.md)
- [Evaluation Metrics](evaluator/README.md)
- [Video Maker with gt/pred actions](visualizer/README.md)
- [Video Feature Extractor](extractor/README.md)
- [Annotation Processor](processor/README.md)

## Running on Local Machine

To run this repository on your local machine, follow the steps below:

1. Install Python 3.10.12
2. Install the required packages by running the following command in the terminal:

    ```bash
    pip install -r requirements.txt
    ```

## Running on Docker with Dev Containers

This repository is configured to run on Docker with Dev Containers. To run this repository on Docker, follow the steps below:

1. Install Docker on your machine.
2. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension on Visual Studio Code.
3. Open this repository on Visual Studio Code as a Dev Container.
