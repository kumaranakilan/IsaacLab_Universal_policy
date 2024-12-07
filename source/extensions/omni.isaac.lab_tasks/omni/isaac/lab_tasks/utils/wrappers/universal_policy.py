# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure a :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv` instance to Stable-Baselines3 vectorized environment.

The following example shows how to wrap an environment for Stable-Baselines3:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper

    env = Sb3VecEnvWrapper(env)

"""

# needed to import for allowing type-hinting: torch.Tensor | dict[str, torch.Tensor]
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn  # noqa: F401
from typing import Any

# TODO: import tdmpc2 library. copy sb3 wrapper


from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

"""
Configuration Parser.
"""
# TODO: copy the on policy runner not sb3
# TODO: make sure the actions are constrained between -1 and +1. tdmpc2 expects this

def process_tdmpc2_cfg(cfg: dict) -> dict:


    pass
# TODO: why do we need a wrapper on both sides??? we can just have one here right?
class UniversalPolicyWrapper():

    pass