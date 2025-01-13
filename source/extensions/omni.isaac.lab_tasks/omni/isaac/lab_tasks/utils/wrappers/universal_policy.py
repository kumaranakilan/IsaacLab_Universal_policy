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

from rsl_rl.env import VecEnv

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

# TODO: Only keep this wrapper and get rid of the other one
# TODO: include the super class
# TODO: delete tdmpc2/envs/wrappers/issac_lab_wrapper.py as it is redundant with the current file

# TODO: probably move this file and the config file in this folder to a subfolder similar to the rsl_rl structure
# NOTE: ActionDTypeWrapper doesn't need to be copied over since the datatype is float32
# NOTE: ActionRepeatWrapper doesn't need to be copied over since we are not repeating actions.
# TODO: Since we are not repeating actions do we need to increase H?
# NOTE: tdmpc2 TimeStepToGymWrapper doesn't need to be copied over since the observation_space, action_space and max_episode_steps are all handled by the IsaacLab env
# NOTE: ExtendedTimeStepWrapper doesn't need to be copied over since it unwraps to action repeater.

# TODO: just because an operation is on torch does not mean it is on GPU. Make sure that all new tensors upon creation are on the same GPU.

# TODO: which logger are we using and are we using wandb

# NOTE this wrapper is a mix of RslRlVecEnvWrapper and TensorWrapper
class UniversalPolicyWrapper(VecEnv):
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv):
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = 0
        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    """
    Properties
    """

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space
    

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}

    def rand_act(self):
        # NOTE: IsaacLab env expects a torch tensor as action input
        # TODO: also find out why action_space.sample() gives a random numpy array
        # NOTE: IsaacLab runs on 32 bit single precision so env.step  expects a float32 torch tensor
        # TODO: How do we handle the unbounded action space
        actions = torch.from_numpy(self.action_space.sample())
        return self._try_f32_tensor(actions)
	
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
    
    def step(self, actions: torch.Tensor):
        # TODO: check if the line below is needed
        assert actions.dtype == torch.float32
        # TODO: this is where the -1, +1 clamping of tdmpc2 should be handled. Also make sure that if the expected action is between -1 and +1 the output action is in the same range
        # TODO: copy from the TensorWrapper code but be careful.
        # NOTE: ./rsl_rl/vecenv_wrapper.py enters a torch tensor in the step function 
        obs_dict, rew, terminated, truncated, extras = self.env.step(action=actions)
        # NOTE: You do not need to borrow the info variable because it is only used to calculate success. There is no episode success in this env.
        # TODO: make sure you are using self._try_f32_tensor correctly below because it is custom code 
        # return self._obs_to_tensor(obs), self._try_f32_tensor(reward), self._try_f32_tensor(done)
        # TODO: claculate the correct definitions of the variables below. They are just a place holder for now
        # NOTE: do not copy the line 'if not self.unwrapped.cfg.is_finite_horizon:' from IsaacLab's vecenv_wrapper because we need access to both pieces of info for now. this might change
        return obs_dict["policy"], rew, terminated, truncated, extras

    def max_episode_length(self):
        return self.env.max_episode_length