#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.


import hydra
import torch
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar
from torch import nn as nn
from vc_models.models.compression_layer import create_compression_layer


class VisualEncoder(nn.Module):
    def __init__(
        self,
        backbone_config: str,
        input_channels: int = 3,
        image_size: int = 128,
        normalize_visual_inputs: bool = True,
        use_augmentations: bool = False,
        loaded_backbone_data = None,
        compression_kernel_size: int = 3,
    ):
        super().__init__()

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(input_channels)
        else:
            self.running_mean_and_var = nn.Sequential()

        backbone_config.defrost()
        backbone_config.transform.resize_size = image_size
        if hasattr(backbone_config.transform, "output_size"):
            backbone_config.transform.output_size = image_size
        if use_augmentations is False:
            if hasattr(backbone_config.transform, "jitter"):
                backbone_config.transform.jitter = False
            if hasattr(backbone_config.transform, "shift"):
                backbone_config.transform.shift = False
        backbone_config.freeze()

        if "resnet" in backbone_config.metadata.model:
            backbone_config.defrost()
            backbone_config.model.use_avgpool_and_flatten = False
            backbone_config.freeze()

        elif "vit" in backbone_config.metadata.model:
            backbone_config.defrost()
            if "model" in backbone_config.model:
                model = backbone_config.model.model
            else:
                model = backbone_config.model

            model.img_size = image_size
            backbone_config.freeze()

        elif "diffusion" in backbone_config.metadata.algo:
            backbone_config.defrost()
            backbone_config.model.tokenize_captions = False
            backbone_config.model.flatten = False
            backbone_config.model.input_image_size = image_size
            backbone_config.freeze()

        elif "vqvae" in backbone_config.metadata.model:
            backbone_config.defrost()
            backbone_config.model.input_image_size = image_size
            backbone_config.freeze()
        else:
            raise ValueError(f"unknown backbone {backbone_config.metadata.model}")

        if loaded_backbone_data is None:
            (
                self.backbone,
                self.embed_dim,
                self.visual_transform,
                _,
            ) = hydra.utils.call(backbone_config)
        else:
            (
                self.backbone,
                self.embed_dim,
                self.visual_transform,
            ) = loaded_backbone_data

        if "resnet" in backbone_config.metadata.model:
            final_spatial_compress = 1.0 / (2**5)
            final_spatial = int(image_size * final_spatial_compress)
        elif "vit" in backbone_config.metadata.model or \
                "vqvae" in backbone_config.metadata.model or \
                "diffusion" in backbone_config.metadata.algo:
            final_spatial = self.backbone.final_spatial

        self.compression, _, self.output_size = create_compression_layer(
            self.embed_dim,
            final_spatial,
            kernel_size=compression_kernel_size
        )

    def get_loaded_backbone_data(self):
        return self.backbone, self.embed_dim, self.visual_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.running_mean_and_var(x)
        # since this is the most expensive part, we do it in minibatches
        output = []
        batch_size = x.shape[0]
        num_mini_batches = 1
        for i in range(num_mini_batches):
            mini_x = x[i * batch_size // num_mini_batches : (i + 1) * batch_size // num_mini_batches]
            mini_x = self.backbone(mini_x).to(torch.float32)
            output.append(mini_x)
        x = torch.cat(output, dim=0)
        x = self.compression(x)
        return x
