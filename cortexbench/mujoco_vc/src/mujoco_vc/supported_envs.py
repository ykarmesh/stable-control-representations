#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import gym

ENV_TO_SUITE = {
    "assembly-v2-goal-observable": "metaworld",
    "bin-picking-v2-goal-observable": "metaworld",
    "button-press-topdown-v2-goal-observable": "metaworld",
    "drawer-open-v2-goal-observable": "metaworld",
    "hammer-v2-goal-observable": "metaworld",
}

SUPPORTED_ENVS = ENV_TO_SUITE.keys()
SUPPORTED_SUITES = ["metaworld"]

DEFAULTS_PROPRIO = {
    "metaworld": None,
}

DEFAULTS_CAMERA = {
    "metaworld": "top_cap2",
}


if __name__ == "__main__":
    # import the suites
    import mj_envs
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    from collections import namedtuple

    for id in ENV_TO_SUITE.keys():
        print("Creating env : %s" % id)
        if ENV_TO_SUITE[id] == "metaworld":
            e = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[id]()
            e._freeze_rand_vec = False
            e.spec = namedtuple("spec", ["id", "max_episode_steps"])
            e.spec.id = id
            e.spec.max_episode_steps = 500
        else:
            e = gym.make(id)
        e.reset()
