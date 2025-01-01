#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torchvision.transforms as T

from vc_models.transforms.to_tensor_if_not import ToTensorIfNot
from vc_models.transforms.random_shifts_aug import RandomShiftsAug
from vc_models.transforms.randomize_env_transform import RandomizeEnvTransform


def diffusion_transforms(resize_size=256, center_crop=True):
    # Preprocessing the datasets.
    return T.Compose(
        [
            T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(resize_size) if center_crop else T.Identity(),
            ToTensorIfNot(),
            T.Normalize([0.5], [0.5]),
        ]
    )


def vit_transforms(resize_size=256, output_size=224):
    return T.Compose(
        [
            T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(output_size),
            ToTensorIfNot(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def resnet_transforms(resize_size=256, output_size=224):
    return T.Compose(
        [
            T.Resize(resize_size),
            T.CenterCrop(output_size),
            ToTensorIfNot(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def transform_augment(
    # Resize/crop
    resize_size=256,
    output_size=224,
    # Jitter
    jitter=True,
    jitter_prob=1.0,
    jitter_brightness=0.3,
    jitter_contrast=0.3,
    jitter_saturation=0.3,
    jitter_hue=0.3,
    # Shift
    shift=True,
    shift_pad=4,
    # Randomize environments
    randomize_environments=False,
    normalize=False,
):
    transforms = [ToTensorIfNot(), T.Resize(resize_size), T.CenterCrop(output_size)]

    if jitter:
        transforms.append(
            T.RandomApply(
                [
                    T.ColorJitter(
                        jitter_brightness,
                        jitter_contrast,
                        jitter_saturation,
                        jitter_hue,
                    )
                ],
                p=jitter_prob,
            )
        )

    if shift:
        transforms.append(RandomShiftsAug(shift_pad))
    
    if normalize:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    transforms = T.Compose(transforms)

    return RandomizeEnvTransform(
        transforms, randomize_environments=randomize_environments
    )