#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image
import hydra
import torch
import numpy as np


# ===================================
# Model Loading
# ===================================
def load_pretrained_model(embedding_config, input_type=np.ndarray, *args, **kwargs):
    """
    Load the pretrained model based on the config corresponding to the embedding_name
    """
    model, embedding_dim, transforms, metadata = hydra.utils.call(embedding_config)
    print("embedding_dim: %s" % embedding_dim)
    model = model.eval()  # model loading API is unreliable, call eval to be double sure

    def final_transforms(transforms):
        if input_type == np.ndarray:
            return lambda input: transforms(Image.fromarray(input)).unsqueeze(0)
        else:
            return transforms

    return model, embedding_dim, final_transforms(transforms), metadata


# ===================================
# Temporal Embedding Fusion
# ===================================
def fuse_embeddings_concat(embeddings: list):
    assert type(embeddings[0]) == np.ndarray
    return np.array(embeddings).ravel()


def fuse_embeddings_flare(embeddings: list):
    if type(embeddings[0]) == np.ndarray:
        history_window = len(embeddings)
        delta = [embeddings[i + 1] - embeddings[i] for i in range(history_window - 1)]
        delta.append(embeddings[-1].copy())
        return np.array(delta).ravel()
    elif type(embeddings[0]) == torch.Tensor:
        history_window = len(embeddings)
        # each embedding will be (Batch, Dim)
        delta = [embeddings[i + 1] - embeddings[i] for i in range(history_window - 1)]
        delta.append(embeddings[-1])
        return torch.cat(delta, dim=1)
    else:
        print("Unsupported embedding format in fuse_embeddings_flare.")
        print("Provide either numpy.ndarray or torch.Tensor.")
        quit()
