# Habitat ImageNav

1. To run experiments in Habitat, first we need to get access to the necessary scene dataset. We are using Gibson scene datasets for our ImageNav experiment. You can find instructions for downloading the dataset [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets).

1. Next we need the episode dataset for ImageNav. You can get the training and validation dataset from [here](https://huggingface.co/datasets/ykarmesh/imagenav_gibson) and place the `train` and `val` folders in the [./data](./data) folder under the path : `data/datasets/pointnav/gibson/v1/`. 

1. Now we are ready to start training the agent. Checkout the `run_habitat_vc.py` script, which allows running an experiment on the cluster. The script can be used as follows:
   ```bash
   python run_habitat_vc.py --config-name=config_imagenav -m 
   ```
   This will start a run on the slurm with the default folder name `imagenav_run`.

1. If you want to start a local run, add `hydra/launcher=slurm` at the end of the command listed in the previous point.

1. Once you have trained a model, it is time for evaluation. We evaluate every 5th saved checkpoint. To run an evaluation, do the following:
   ```bash
   python run_habitat_vc.py --config-name=eval_config_imagenav hydra/launcher=slurm_eval NUM_ENVIRONMENTS=14 -m
   ```
