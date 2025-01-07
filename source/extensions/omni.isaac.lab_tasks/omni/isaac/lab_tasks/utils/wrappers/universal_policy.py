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

from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn  # noqa: F401
from typing import Any

# TODO: import tdmpc2 library. copy sb3 wrapper

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

# TODO: Only keep this wrapper and get rid of the other one
# TODO: include the super class
# TODO: delete tdmpc2/envs/wrappers/issac_lab_wrapper.py as it is redundant with the current file

# NOTE: ActionDTypeWrapper doesn't need to be copied over since the datatype is float32
# NOTE: ActionRepeatWrapper doesn't need to be copied over since we are not repeating actions.
# TODO: Since we are not repeating actions do we need to increase H?
# NOTE: tdmpc2 TimeStepToGymWrapper doesn't need to be copied over since the observation_space, action_space and max_episode_steps are all handled by the IsaacLab env
# NOTE: ExtendedTimeStepWrapper doesn't need to be copied over since it unwraps to action repeater.


# NOTE this wrapper is a mix of RslRlVecEnvWrapper and TensorWrapper
class UniversalPolicyWrapper():
    def __init__(self, env):
        super().__init__(env)

    def rand_act(self):
        # NOTE: IsaacLab env expects a torch tensor as action input
        # TODO: also find out why action_space.sample() gives a random numpy array
        # NOTE: IsaacLab runs on 32 bit single precision so env.step  expects a float32 torch tensor
        # TODO: How do we handle the unbounded action space
        return torch.from_numpy(self.action_space.sample().astype(np.float32))
	
    def _try_f32_tensor(self, x: torch.Tensor):
        if x.dtype == torch.float64:
            x = x.float()
        return x

    def _obs_to_tensor(self, obs: torch.Tensor):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_tensor(obs[k])
        else:
            obs = self._try_f32_tensor(obs)
        return obs

    def reset(self, dones: torch.Tensor = None):
        # NOTE: The current env doesn't require POMDP. This logic is in the doc.
        # NOTE: ManagerBasedRLEnv handles the resetting in the step function. This only reset 
        return self._obs_to_tensor(self.env.reset())
    
    def step(self, action: torch.Tensor):
        # TODO: check if the line below is needed
        assert action.dtype == torch.float32
        # TODO: this is where the -1, +1 clamping of tdmpc2 should be handled
        # TODO: copy from the TensorWrapper code but be careful.
        # NOTE: ./rsl_rl/vecenv_wrapper.py enters a torch tensor in the step function 
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # TODO: what is info and what is sucess. search this in tdmpc2 and check if it is in extras
        info = defaultdict(float, info)
        info['success'] = float(info['success'])
        # TODO: make sure you are using self._try_f32_tensor correctly below because it is custom code 
        # return self._obs_to_tensor(obs), self._try_f32_tensor(reward), self._try_f32_tensor(done), info
        # TODO: claculate the correct definitions of the variables below. They are just a place holder for now
        # NOTE: do not copy the line 'if not self.unwrapped.cfg.is_finite_horizon:' from vecenv_wrapper because we need access to both pieces of info for now. this might change
        return obs_dict, rew, terminated, truncated, extras