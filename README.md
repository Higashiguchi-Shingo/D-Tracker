## D-Tracker

Implementation of "D-Tracker: Modeling Interest Diffusion in Social Activity Tensor Data Streams." (KDD2025)

[Paper link: https://dl.acm.org/doi/abs/10.1145/3690624.3709192]


## Install

Using pip:

```bash
pip install -r requirements.txt
```

## Dataset

All datasets used in the paper are located in the `data/` directory.

## Running Experiments

Scripts for running experiments are provided in the `bin/` directory.

Please execute `run.sh` with the dataset name as an argument:

```bash
cd bin
sh run.sh <dataset-name>
```