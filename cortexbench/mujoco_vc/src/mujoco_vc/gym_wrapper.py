#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np, torch
import gym
from mjrl.utils.gym_env import GymEnv
from gym.spaces.box import Box
from mujoco_vc.model_loading import load_pretrained_model
from mujoco_vc.supported_envs import ENV_TO_SUITE, SUPPORTED_SUITES
from mujoco_vc.supported_envs import DEFAULTS_PROPRIO
from typing import Union


class MuJoCoPixelObsWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        width,
        height,
        camera_name,
        device_id=-1,
        depth=False,
        *args,
        **kwargs,
    ):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0.0, high=255.0, shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id
        self.suite = ENV_TO_SUITE[self.spec.id]

        self.get_image = lambda: get_image(
            env=self,
            suite=self.suite,
            camera_name=self.camera_name,
            height=self.height,
            width=self.width,
            depth=self.depth,
            device_id=self.device_id,
        )

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        # Output format is (H, W, 3)
        return self.get_image()


class FrozenEmbeddingWrapper(gym.ObservationWrapper):
    """
    This wrapper places a frozen vision model over the image observation.

    Args:
        env (Gym environment): the original environment
        suite (str): category of environment ["metaworld"]
        embedding_name (str): name of the embedding to use (name of config)
        history_window (int, 1) : timesteps of observation embedding to incorporate into observation (state)
        embedding_fusion (callable, 'None'): function for fusing the embeddings into a state.
            Defaults to concatenation if not specified
        obs_dim (int, 'None') : dimensionality of observation space. Inferred if not specified.
            Required if function != None. Defaults to history_window * embedding_dim
        add_proprio (bool, 'False') : flag to specify if proprioception should be appended to observation
        device (str, 'cuda'): where to allocate the model.
    """

    def __init__(
        self,
        env,
        embedding_config: dict,
        suite: str,
        history_window: int = 1,
        fuse_embeddings: callable = None,
        obs_dim: int = None,
        device: str = "cuda",
        seed: int = None,
        add_proprio: bool = False,
        proprio_keys: list = None,
        *args,
        **kwargs,
    ):
        gym.ObservationWrapper.__init__(self, env)
        self.embedding_buffer = (
            []
        )  # buffer to store raw embeddings of the image observation
        self.obs_buffer = []  # temp variable, delete this line later
        self.history_window = history_window
        self.fuse_embeddings = fuse_embeddings
        if device == "cuda" and torch.cuda.is_available():
            print("Using CUDA.")
            device = torch.device("cuda")
        else:
            print("Not using CUDA.")
            device = torch.device("cpu")
        self.device = device

        # get the embedding model
        embedding, embedding_dim, transforms, metadata = load_pretrained_model(
            embedding_config=embedding_config, seed=seed
        )
        embedding.to(device=self.device)
        # freeze the PVR
        for p in embedding.parameters():
            p.requires_grad = False
        self.embedding, self.embedding_dim, self.transforms = (
            embedding,
            embedding_dim,
            transforms,
        )

        # proprioception
        if add_proprio:
            self.get_proprio = lambda: get_proprioception(
                self.unwrapped, suite, proprio_keys
            )
            proprio = self.get_proprio()
            self.proprio_dim = 0 if proprio is None else proprio.shape[0]
        else:
            self.proprio_dim = 0
            self.get_proprio = None

        # final observation space
        obs_dim = (
            obs_dim
            if obs_dim != None
            else int(self.history_window * self.embedding_dim + self.proprio_dim)
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

    def observation(self, observation):
        # observation shape : (H, W, 3)
        inp = self.transforms(
            observation
        )  # numpy to PIL to torch.Tensor. Final dimension: (1, 3, H, W)
        inp = inp.to(self.device)
        with torch.no_grad():
            emb = (
                self.embedding(inp)
                .view(-1, self.embedding_dim)
                .to("cpu")
                .numpy()
                .squeeze()
            )
        # update observation buffer
        if len(self.embedding_buffer) < self.history_window:
            # initialization
            self.embedding_buffer = [emb.copy()] * self.history_window
        else:
            # fixed size buffer, replace oldest entry
            for i in range(self.history_window - 1):
                self.embedding_buffer[i] = self.embedding_buffer[i + 1].copy()
            self.embedding_buffer[-1] = emb.copy()

        # fuse embeddings to obtain observation
        if self.fuse_embeddings != None:
            obs = self.fuse_embeddings(self.embedding_buffer)
        else:
            # print("Fuse embedding function not given. Defaulting to concat.")
            obs = np.array(self.embedding_buffer).ravel()

        # add proprioception if necessary
        if self.proprio_dim > 0:
            proprio = self.get_proprio()
            obs = np.concatenate([obs, proprio])
        return obs

    def get_obs(self):
        return self.observation(self.env.observation(None))

    def get_image(self):
        return self.env.get_image()

    def reset(self):
        self.embedding_buffer = []  # reset to empty buffer
        return super().reset()


def env_constructor(
    env_name: str,
    embedding_config: dict,
    pixel_based: bool = True,
    device: str = "cuda",
    image_width: int = 256,
    image_height: int = 256,
    camera_name: str = None,
    embedding_name: str = "resnet50",
    history_window: int = 1,
    fuse_embeddings: callable = None,
    render_gpu_id: int = -1,
    seed: int = 123,
    add_proprio=False,
    proprio_keys=None,
    *args,
    **kwargs,
) -> GymEnv:
    # construct basic gym environment
    assert env_name in ENV_TO_SUITE.keys()
    suite = ENV_TO_SUITE[env_name]
    if suite == "metaworld":
        # Meta world natively misses many specs. We will explicitly add them here.
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        from collections import namedtuple

        e = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
        e._freeze_rand_vec = False
        e.spec = namedtuple("spec", ["id", "max_episode_steps"])
        e.spec.id = env_name
        e.spec.max_episode_steps = 500
    else:
        raise NotImplementedError("Only metaworld environments are supported.")
    # seed the environment for reproducibility
    e.seed(seed)

    # get correct camera name
    camera_name = (
        None if (camera_name == "None" or camera_name == "default") else camera_name
    )
    # Use appropriate observation wrapper
    if pixel_based:
        e = MuJoCoPixelObsWrapper(
            env=e,
            width=image_width,
            height=image_height,
            camera_name=camera_name,
            device_id=0,
        )
        e = FrozenEmbeddingWrapper(
            env=e,
            embedding_config=embedding_config,
            suite=suite,
            history_window=history_window,
            fuse_embeddings=fuse_embeddings,
            device=device,
            seed=seed,
            add_proprio=add_proprio,
            proprio_keys=proprio_keys,
        )
        e = GymEnv(e)
    else:
        e = GymEnv(e)

    # Output wrapped env
    e.set_seed(seed)
    return e


def get_proprioception(
    env: gym.Env,
    suite: str,
    proprio_keys: Union[list, None, str] = "default",  # only for robohive
    *args,
    **kwargs,
) -> Union[np.ndarray, None]:
    # Checks + Default behaviors
    assert isinstance(env, gym.Env)
    assert suite in SUPPORTED_SUITES
    if proprio_keys == "default":
        proprio_keys = DEFAULTS_PROPRIO[suite]

    if suite == "metaworld":
        return env.unwrapped._get_obs()[:4]
    else:
        print("Unsupported environment. Proprioception is defaulting to None.")
        return None


def get_image(
    env: gym.Env,
    suite: str,
    camera_name: str,
    height: int = 224,
    width: int = 224,
    depth: bool = False,
    device_id: int = -1,
) -> Union[np.ndarray, None]:
    assert isinstance(env, gym.Env)
    if suite == "metaworld":
        # mujoco-py backend
        img = env.unwrapped.sim.render(
            width=width,
            height=height,
            depth=depth,
            camera_name=camera_name,
            device_id=device_id,
        )
        img = img[::-1, :, :]
    else:
        print("Unsupported Env Suite.")
        assert (
            suite in SUPPORTED_SUITES
        ), f"{suite} not in supported suites {SUPPORTED_SUITES}"
    return img
