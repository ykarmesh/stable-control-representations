from mujoco_vc.model_loading import (
    load_pretrained_model,
    fuse_embeddings_flare,
)
from torch.utils.data import Dataset
from tqdm import tqdm
import gym, mjrl.envs, robohive
import pickle, torch, glob, h5py
import numpy as np

class FrozenVILDataset(Dataset):
    """
    Dataset for visual imitation learning with frozen visual backbone.
    This class loads a provided dataset, pre-computes features with a
    frozen visual model and only retains the pre-computed features.
    Supports dataset in both pickle and h5 formats.
    """

    def __init__(
        self,
        embeddings_config: str,
        data_path: str,
        data_type: str = "list",
        history_window: int = 1,
        chunk_size: int = 20,
        fuse_embeddings: callable = fuse_embeddings_flare,
        vision_key: str = "images",
        # vision_key can also take the form "env_infos/rgb:top_cam:256x256:2d"
        # for now only support a single key (i.e. single viewpoint/camera)
        proprio_keys: list = [],
        device: str = "cuda",
        max_paths: int = None,
        diffusion_prompt: str = None,
        diffusion_timesteps: int = None,
        remove_last_step: bool = False,
        *args,
        **kwargs,
    ):
        """
        Args:
            embeddings_config:  config of the visual embedding model
            data_path:           path to the dataset
            data_type:          "list" or "h5"
            history_window:     number of previous frames to use as context
            chunk_size:         number of frames to process at a time
            fuse_embeddings:    function to fuse embeddings
            vision_key:         key to access vision data in the dataset e.g. "images",
                                  the key can also be nested and take the form
                                  "env_infos/rgb:top_cam:256x256:2d" where each '/'
                                  represents a nested dictionary
            proprio_keys:       list of keys to access proprioceptive data in the dataset,
                                  e.g. ["actions", "env_infos/robot-state"]
            device:             "cuda" or "cpu"
            max_paths:          maximum number of paths to load from the dataset
            diffusion_prompt:   prompt to use for diffusion
            diffusion_timesteps: number of diffusion timesteps
            remove_last_step:   whether to remove the last step of the trajectory

        """
        self.data_path = data_path
        self.data_type = data_type
        self.vision_key = vision_key
        self.proprio_keys = proprio_keys
        self.device = device
        self.fuse_embeddings = fuse_embeddings
        self.embeddings_config = embeddings_config
        self.history_window = history_window
        self.chunk_size = chunk_size
        self.max_paths = max_paths
        self.diffusion_prompt = diffusion_prompt
        self.diffusion_timesteps = diffusion_timesteps
        self.remove_last_step = remove_last_step


        # Get embeddings and features
        paths, self.embeddings, self.features = self._preprocess_paths()

        # Get actions
        self.actions = []
        for path in paths:
            if self.remove_last_step:
                self.actions.append([actions for actions in path["actions"][:-1]])
            else:
                self.actions.append([actions for actions in path["actions"]])

        self.path_length = max([(p["actions"].shape[0] - 1*self.remove_last_step) for p in paths])
        self.num_paths = len(paths)

        # Release memory
        del paths

    def _load_dataset(self):
        if self.data_type == "list":
            if len(self.proprio_keys) > 0:
                self.proprio_keys = [f"env_infos/{k}" for k in self.proprio_keys]
            return pickle.load(open(self.data_path, "rb"))
        elif self.data_type == "h5":
            self.vision_key = f"env_infos/visual_dict/{self.vision_key}"
            if len(self.proprio_keys) > 0:
                self.proprio_keys = [
                    f"env_infos/proprio_dict/{k}" for k in self.proprio_keys
                ]
            return h5_to_paths(self.data_path)
        raise Exception("data_type must be either 'list' or 'h5'")

    def _extract_nested_key(self, dic, key):
        keys = key.split("/")
        for k in keys:
            if k=="ee_pose_wrt_robot" and "ee_pose" in dic.keys():
                dic = dic["ee_pose"]
                continue
            dic = dic[k]
        
        return dic[:-self.remove_last_step] if self.remove_last_step else dic

    def _compute_embeddings(self, paths, chunk_size):
        """
        Compute embeddings for every image in every path in the dataset.
        """
        model, embedding_dim, transforms, metadata = load_pretrained_model(
            self.embeddings_config
        )
        model = model.to(self.device)

        all_embeddings = []
        for path in tqdm(paths):
            new_noise = True
            inp = self._extract_nested_key(path, self.vision_key)
            embeddings = np.zeros((inp.shape[0], embedding_dim))
            path_len = inp.shape[0]


            # shape (B, 3, H, W)
            preprocessed_inp = torch.cat([transforms(frame) for frame in inp])
            for chunk in range(path_len // chunk_size + 1):
                if chunk_size * chunk < path_len:
                    with torch.no_grad():
                        inp_chunk = preprocessed_inp[
                            chunk_size * chunk : min(chunk_size * (chunk + 1), path_len)
                        ]

                        if self.diffusion_prompt is not None:
                            emb = model(inp_chunk.to(self.device), self.diffusion_prompt, self.diffusion_timesteps, new_noise)
                            new_noise = False
                        else:
                            emb = model(inp_chunk.to(self.device))

                        # save embedding in RAM and free up GPU memory
                        emb = emb.to("cpu").data.numpy()

                    embeddings[
                        chunk_size * chunk : min(chunk_size * (chunk + 1), path_len)
                    ] = emb

            # no longer need the images or model, free up RAM
            # del path["images"]
            all_embeddings.append(embeddings)

        # Release memory
        del model
        del transforms
        return all_embeddings

    def _compute_features(self, paths, all_embeddings, history_window):
        """
        Compute features for every image in every path in the dataset.
        The features are computed by fusing the embeddings of the current
        frame and the previous history_window frames. And then concatenating
        the features with the proprioceptive data.
        """
        all_features = []
        for k, path in enumerate(paths):
            embeddings = all_embeddings[k]
            features = []
            for t in range(embeddings.shape[0]):
                emb_hist_t = [embeddings[max(t - k, 0)] for k in range(history_window)]
                emb_hist_t = emb_hist_t[
                    ::-1
                ]  # emb_hist_t[-1] should correspond to time t embedding
                feat_t = self.fuse_embeddings(emb_hist_t)

                if len(self.proprio_keys) > 0:
                    proprio_data = []
                    for proprio_key in self.proprio_keys:
                        proprio = self._extract_nested_key(path, proprio_key)
                        proprio_data.append(proprio[t])

                    proprio_data = np.concatenate(proprio_data)
                    feat_t = np.concatenate([feat_t, proprio_data])

                features.append(feat_t.copy())

            all_features.append(np.array(features))
        return all_features

    def _preprocess_paths(self):
        """
        Loads the dataset and the vision model, precomputes the embeddings and
        fuses them with the proprioception to form features. Returns the paths
        """
        paths = self._load_dataset()
        if self.max_paths is not None:
            paths = paths[: self.max_paths]
        embeddings = self._compute_embeddings(paths, self.chunk_size)
        features = self._compute_features(paths, embeddings, self.history_window)
        return paths, embeddings, features

    def __len__(self):
        return self.path_length * self.num_paths

    def __getitem__(self, index):
        traj_idx = int(index // self.path_length)
        timestep = int(index - traj_idx * self.path_length)
        timestep = min(timestep, len(self.actions[traj_idx]))

        features = self.features[traj_idx][timestep]
        action = self.actions[traj_idx][timestep]

        return {"features": features, "actions": action}


class FrozenEmbeddingDataset(Dataset):
    def __init__(
        self,
        paths: list,
        history_window: int = 1,
        fuse_embeddings: callable = None,
        device: str = "cuda",
    ):
        self.paths = paths
        assert "embeddings" in self.paths[0].keys()
        # assume equal length trajectories
        # code will work even otherwise but may have some edge cases
        self.path_length = max([p["actions"].shape[0] for p in paths])
        self.num_paths = len(self.paths)
        self.history_window = history_window
        self.fuse_embeddings = fuse_embeddings
        self.device = device

    def __len__(self):
        return self.path_length * self.num_paths

    def __getitem__(self, index):
        traj_idx = int(index // self.path_length)
        timestep = int(index - traj_idx * self.path_length)
        timestep = min(timestep, self.paths[traj_idx]["actions"].shape[0])
        if "features" in self.paths[traj_idx].keys():
            features = self.paths[traj_idx]["features"][timestep]
            action = self.paths[traj_idx]["actions"][timestep]
        else:
            embeddings = [
                self.paths[traj_idx]["embeddings"][max(timestep - k, 0)]
                for k in range(self.history_window)
            ]
            embeddings = embeddings[
                ::-1
            ]  # embeddings[-1] should be most recent embedding
            features = self.fuse_embeddings(embeddings)
            # features = torch.from_numpy(features).float().to(self.device)
            action = self.paths[traj_idx]["actions"][timestep]
            # action   = torch.from_numpy(action).float().to(self.device)
        return {"features": features, "actions": action}


def compute_embeddings(
    paths: list, embeddings_config: str, device: str = "cpu", chunk_size: int = 20
):
    model, embedding_dim, transforms, metadata = load_pretrained_model(
        embeddings_config
    )
    model.to(device)
    for path in tqdm(paths):
        inp = path["images"]  # shape (B, H, W, 3)
        path["embeddings"] = np.zeros((inp.shape[0], embedding_dim))
        path_len = inp.shape[0]
        preprocessed_inp = torch.cat(
            [transforms(frame) for frame in inp]
        )  # shape (B, 3, H, W)
        for chunk in range(path_len // chunk_size + 1):
            if chunk_size * chunk < path_len:
                with torch.no_grad():
                    inp_chunk = preprocessed_inp[
                        chunk_size * chunk : min(chunk_size * (chunk + 1), path_len)
                    ]
                    emb = model(inp_chunk.to(device))
                    # save embedding in RAM and free up GPU memory
                    emb = emb.to("cpu").data.numpy()
                path["embeddings"][
                    chunk_size * chunk : min(chunk_size * (chunk + 1), path_len)
                ] = emb
        del path["images"]  # no longer need the images, free up RAM
    return paths


def precompute_features(
    paths: list,
    history_window: int = 1,
    fuse_embeddings: callable = None,
    proprio_key: str = None,
):
    assert "embeddings" in paths[0].keys()
    for path in paths:
        features = []
        for t in range(path["embeddings"].shape[0]):
            emb_hist_t = [
                path["embeddings"][max(t - k, 0)] for k in range(history_window)
            ]
            emb_hist_t = emb_hist_t[
                ::-1
            ]  # emb_hist_t[-1] should correspond to time t embedding
            feat_t = fuse_embeddings(emb_hist_t)
            if proprio_key not in [None, "None", []]:
                assert proprio_key in path["env_infos"].keys()
                feat_t = np.concatenate([feat_t, path["env_infos"][proprio_key][t]])
            features.append(feat_t.copy())
        path["features"] = np.array(features)
    return paths


def h5_to_paths(data_root: str) -> list:
    """
    Convert H5 files from a provided data_root into the paths format
    (i.e. list of dictionaries) that is typically used in RL/IL.
    Assumed file format is: f"{data_root}/seed{idx}/*_trace.h5"
    """
    files = glob.glob(data_root + "/*/*_trace.h5")
    subsets = [h5py.File(path, "r") for path in files]
    h5_paths = [subset_paths[p] for subset_paths in subsets for p in subset_paths]
    return h5_paths
