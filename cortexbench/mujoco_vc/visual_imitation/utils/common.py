import os, torch
import numpy as np


def set_seed(seed=None):
    """
    Set all seeds to make results reproducible
    :param seed: an integer to your choosing (default: None)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


def configure_cluster_GPUs(gpu_logical_id: int) -> int:
    """
    Maps the GPU logical ID to physical ID. This is required for MuJoCo to
    correctly use the GPUs, since it relies on physical ID unlike pytorch
    """
    # get the correct GPU ID
    if "SLURM_STEP_GPUS" in os.environ.keys():
        physical_gpu_ids = os.environ.get("SLURM_STEP_GPUS")
        gpu_id = int(physical_gpu_ids.split(",")[gpu_logical_id])
        print("Found slurm-GPUS: <Physical_id:{}>".format(physical_gpu_ids))
        print(
            "Using GPU <Physical_id:{}, Logical_id:{}>".format(gpu_id, gpu_logical_id)
        )
    else:
        gpu_id = 0  # base case when no GPUs detected in SLURM
        print("No GPUs detected. Defaulting to 0 as the device ID")
    return gpu_id
