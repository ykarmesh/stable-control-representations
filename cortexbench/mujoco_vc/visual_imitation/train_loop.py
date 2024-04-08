#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.core import sample_paths
from mjrl.policies.gaussian_mlp import MLP, BatchNormConvPoolMLP
from mjrl.algos.behavior_cloning import BC
from mujoco_vc.gym_wrapper import env_constructor
from mujoco_vc.rollout_utils import rollout_from_init_states
from mujoco_vc.model_loading import (
    fuse_embeddings_concat,
    fuse_embeddings_flare,
)
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from vc_models.utils.wandb import setup_wandb
from utils.common import set_seed, configure_cluster_GPUs
from utils.dataset_utils import FrozenVILDataset
import gym, mjrl.envs, robohive
import numpy as np, time as timer, pickle, os, torch, gc

def bc_pvr_train_loop(config: dict) -> None:
    # configure GPUs
    # os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    physical_gpu_id = configure_cluster_GPUs(config["env_kwargs"]["render_gpu_id"])
    config["env_kwargs"]["render_gpu_id"] = physical_gpu_id

    # set the seed
    set_seed(config["seed"])

    # infer the demo location
    if config["env_kwargs"]["suite"] in ["dmc", "adroit", "metaworld"]:
        demo_paths_loc = os.path.join(
            config["data_dir"], config["env_kwargs"]["env_name"] + ".pickle"
        )
        data_type = "list"
    elif config["env_kwargs"]["suite"] == "franka_kitchen":
        demo_paths_loc = os.path.join(
            config["data_dir"], config["env_kwargs"]["env_name"]
        )
        data_type = "h5"
    else:
        print("\n\n Unsupported environment suite.")
        quit()

    if data_type == "list":
        try:
            demo_paths = pickle.load(open(demo_paths_loc, "rb"))
        except:
            print("Unable to load the data. Check the data path.")
            print(demo_paths_loc)
            quit()

        demo_paths = demo_paths[: config["num_demos"]]
        demo_score = np.mean([np.sum(p["rewards"]) for p in demo_paths])
        print("Number of demonstrations used : %i" % len(demo_paths))
        print("Demonstration score : %.2f " % demo_score)

    # store init_states for evaluation on training trajectories
    if config["env_kwargs"]["suite"] == "dmc":
        init_states = [
            p["env_infos"]["internal_state"][0].astype(np.float64) for p in demo_paths
        ]
    elif config["env_kwargs"]["suite"] == "adroit":
        init_states = [p["init_state_dict"] for p in demo_paths]
    elif config["env_kwargs"]["suite"] in ["metaworld", "franka_kitchen"]:
        init_states = []
    else:
        print("\n\n Unsupported environment suite.")
        quit()

    # construct the environment and policy
    env_kwargs = config["env_kwargs"]

    if config["emb_fusion"] == "concat":
        fuse_embeddings = fuse_embeddings_concat
    elif config["emb_fusion"] == "flare":
        fuse_embeddings = fuse_embeddings_flare
    else:
        raise Exception("Unsupported embedding fusion method. {}".format(config["emb_fusion"]))

    e = env_constructor(**env_kwargs, fuse_embeddings=fuse_embeddings)
    policy = BatchNormConvPoolMLP(
        env_spec=e.spec,
        hidden_sizes=eval(config["bc_kwargs"]["hidden_sizes"]),
        seed=config["seed"],
        nonlinearity=config["bc_kwargs"]["nonlinearity"],
        dropout=config["bc_kwargs"]["dropout"],
        embedding_consolidation_type=config["emb_consolidation_type"],
        spatial_dims=config["spatial_dims"],
        embed_channel_dim=config["embed_channel_dim"],
        proprio_dim=config["proprio_dim"],
        stride=config["stride"],
        reduced_channel_dim=config["reduced_channel_dim"],
    )

    # compute embeddings and create dataset
    print("===================================================================")
    print(">>>>>>>>> Precomputing frozen embedding dataset >>>>>>>>>>>>>>>>>>>")

    max_paths = None
    if "max_paths" in config:
        max_paths = config["max_paths"]

    dataset = FrozenVILDataset(
        config["env_kwargs"]["embedding_config"],
        demo_paths_loc,
        data_type=data_type,
        history_window=config["env_kwargs"]["history_window"],
        fuse_embeddings=fuse_embeddings,
        vision_key=config["env_kwargs"]["vision_key"],
        proprio_keys=config["env_kwargs"]["proprio_keys"],
        device=config["device"],
        max_paths=max_paths,
        diffusion_prompt=config["diffusion_prompt"] if "diffusion_prompt" in config else None,
        diffusion_timesteps=config["diffusion_timesteps"] if "diffusion_timesteps" in config else None,
        remove_last_step=True if config["env_kwargs"]["suite"] == "franka_kitchen" else False,
    )
    gc.collect()  # garbage collection to free up RAM

    # Dataset in this case is pre-loaded and on the RAM (CPU) and not on the disk
    dataloader = DataLoader(
        dataset,
        batch_size=config["bc_kwargs"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    optimizer = torch.optim.Adam(
        list(policy.parameters()),
        lr=config["bc_kwargs"]["lr"]
    )
    loss_func = torch.nn.MSELoss()

    # Update logging to match CortexBench conventions
    # Make log dir
    wandb_run = setup_wandb(config)
    if os.path.isdir(config["job_name"]) == False:
        os.mkdir(config["job_name"])
    previous_dir = os.getcwd()
    os.chdir(config["job_name"])  # important! we are now in the directory to save data
    if os.path.isdir("iterations") == False:
        os.mkdir("iterations")
    if os.path.isdir("logs") == False:
        os.mkdir("logs")

    highest_tr_score, highest_score = -np.inf, -np.inf
    highest_tr_success, highest_success = 0.0, 0.0
    for epoch in tqdm(range(config["epochs"])):
        # move the policy to correct device
        policy.to(config["device"])
        policy.train()
        # update policy for one BC epoch
        running_loss = 0.0
        for mb_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            feat = batch["features"].float().to(config["device"])
            # project the feat and flatten the last three dimensions
            tar = batch["actions"].float().to(config["device"])
            pred = policy.forward(feat)
            loss = loss_func(pred, tar.detach())
            # add l1  norm on the weights
            if config["bc_kwargs"]["l1_weight"] > 0:
                l1_reg = torch.tensor(0.0, requires_grad=True).to(config["device"])
                for name, param in policy.named_parameters():
                    if "weight" in name:
                        l1_reg = l1_reg + torch.norm(param, 1)
                loss = loss + config["bc_kwargs"]["l1_weight"] * l1_reg
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.to("cpu").data.numpy().ravel()[0]
        # log average loss for the epoch
        wandb_run.log({"epoch_loss": running_loss / (mb_idx + 1)}, step=epoch + 1)


        # perform evaluation rollouts every few epochs
        if (epoch % config["eval_frequency"] == 0 and epoch > 0) or (
            epoch == config["epochs"] - 1
        ):
            # move the policy to CPU for saving and evaluation
            policy.output_device_to("cpu")

            policy.eval()
            # ensure enironment embedding is in eval mode before rollouts
            e.env.embedding.eval()

            paths = sample_paths(
                num_traj=config["eval_num_traj"],
                env=e,
                policy=policy,
                eval_mode=True,
                horizon=e.horizon,
                base_seed=config["seed"],
                num_cpu=config["num_cpu"],
                get_images=config["save_video"],
            )
            (
                mean_score,
                success_percentage,
                highest_score,
                highest_success,
            ) = compute_metrics_from_paths(
                env=e,
                suite=config["env_kwargs"]["suite"],
                paths=paths,
                highest_score=highest_score,
                highest_success=highest_success,
            )
            epoch_log = {}
            epoch_log["eval/epoch"] = epoch
            epoch_log["eval/score_mean"] = mean_score
            epoch_log["eval/success"] = success_percentage
            epoch_log["eval/highest_success"] = highest_success
            epoch_log["eval/highest_score"] = highest_score

            # create_video_from_path(paths, config, epoch, "eval")

            # log statistics on training paths
            if init_states is not None and len(init_states) > 0:
                paths = rollout_from_init_states(
                    init_states[: config["eval_num_traj"]],
                    e,
                    policy,
                    eval_mode=True,
                    horizon=e.horizon,
                )
            else:
                # use same seed as used for collecting the training paths
                paths = sample_paths(
                    num_traj=config["eval_num_traj"],
                    env=e,
                    policy=policy,
                    eval_mode=True,
                    horizon=e.horizon,
                    base_seed=54321,
                    num_cpu=config["num_cpu"],
                    get_images=config["save_video"],
                )
            (
                tr_score,
                tr_success,
                highest_tr_score,
                highest_tr_success,
            ) = compute_metrics_from_paths(
                env=e,
                suite=config["env_kwargs"]["suite"],
                paths=paths,
                highest_score=highest_tr_score,
                highest_success=highest_tr_success,
            )
            epoch_log["train/epoch"] = epoch
            epoch_log["train/score"] = tr_score
            epoch_log["train/success"] = tr_success
            epoch_log["train/highest_score"] = highest_tr_score
            epoch_log["train/highest_success"] = highest_tr_success

            create_video_from_path(paths, config, epoch, "train")
            # Log with wandb
            wandb_run.log(data=epoch_log)

            print(
                "Epoch = %i | BC performance (eval mode) = %.3f " % (epoch, mean_score)
            )
            print(tabulate(sorted(epoch_log.items())))

        # save policy and logging
        if (epoch % config["save_frequency"] == 0 and epoch > 0) or (
            epoch == config["epochs"] - 1
        ):
            # move the policy to CPU for saving and evaluation
            policy.output_device_to("cpu")

            policy.eval()
            # ensure enironment embedding is in eval mode before rollouts
            e.env.embedding.eval()

            # pickle.dump(agent.policy, open('./iterations/policy_%i.pickle' % epoch, 'wb'))
            if highest_score == mean_score:
                pickle.dump(policy, open("./iterations/best_policy.pickle", "wb"))


def compute_metrics_from_paths(
    env: GymEnv,
    suite: str,
    paths: list,
    highest_score: float = -1.0,
    highest_success: float = -1.0,
):
    mean_score = np.mean([np.sum(p["rewards"]) for p in paths])
    if suite == "dmc":
        # we evaluate dmc based on returns, not success
        success_percentage = -1.0
    elif suite == "adroit":
        success_percentage = env.env.unwrapped.evaluate_success(paths)
    elif suite == "metaworld":
        sc = []
        for i, path in enumerate(paths):
            sc.append(path["env_infos"]["success"][-1])
        success_percentage = np.mean(sc) * 100
    elif suite == "franka_kitchen":
        success_percentage = np.mean(
            [path["env_infos"]["solved"][-1] for path in paths]
        )
    else:
        raise NotImplementedError

    highest_score = mean_score if mean_score >= highest_score else highest_score
    highest_success = (
        success_percentage if success_percentage >= highest_success else highest_success
    )
    return mean_score, success_percentage, highest_score, highest_success

def create_video_from_path(
        paths: list,
        config: dict,
        epoch: int,
        mode: str,
    ):
    # create a gif of the rollout
    import imageio
    if not config["save_video"]:
        return

    for i, path in enumerate(paths):
        images = []
        for image in path["images"]:
            images.append(image)
        imageio.mimsave(
            "./iterations/%s_%i_%i.gif" % (mode, epoch, i),
            images,
            duration=0.02,
        )