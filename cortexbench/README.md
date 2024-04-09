# Benchmarks

Currently we are releasing our codebase with support for the following benchmarks:


| **Benchmark Suite** | **Observation Space** | **Action Space** | **Goal Specification** | **Policy Learning** |
|----------------------|--------------------------|----------------------|---------------------------|-------------------|
| [Metaworld](./mujoco_vc#metaworld-benchmark) | RGB + proprio. | Continuous | - | IL |
| [ImageNav](./habitat_vc#imagenav) | RGB | Discrete | Goal Image | RL |

We are planning to add the OVMM benchmark soon. 

## Installation

To install CortexBench, please follow the instructions in [../INSTALLATION.md](../INSTALLATION.md).

## Datasets

| **Benchmark** | **Description** | **Download Link** |
|---------------|-----------------|--------------------|
| [ImageNav](./habitat_vc#imagenav) | The ImageNav benchmark uses Gibson scene datasets. The training and validation episode datasets for ImageNav can be downloaded from this link. | [Gibson dataset](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets), [Training and Validation dataset](https://huggingface.co/datasets/ykarmesh/imagenav_gibson) |
| [Metaworld](./mujoco_vc#metaworld-benchmark) | The Metaworld benchmark dataset. | [Metaworld dataset](https://dl.fbaipublicfiles.com/eai-vc/mujoco_vil_datasets/metaworld-expert-v1.0.zip) |
