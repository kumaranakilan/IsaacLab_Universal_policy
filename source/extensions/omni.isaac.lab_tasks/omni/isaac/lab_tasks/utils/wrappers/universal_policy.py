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

# TODO:  (Low priority) probably move this file and the config file in this folder to a subfolder similar to the rsl_rl structure
# NOTE: ActionDTypeWrapper doesn't need to be copied over since the datatype is float32
# NOTE: ActionRepeatWrapper doesn't need to be copied over since we are not repeating actions.
# TODO: (Low priority) Since we are not repeating actions do we need to increase H?
# NOTE: tdmpc2 TimeStepToGymWrapper doesn't need to be copied over since the observation_space, action_space and max_episode_steps are all handled by the IsaacLab env
# NOTE: ExtendedTimeStepWrapper doesn't need to be copied over since it unwraps to action repeater.

# TODO: (Low priority) just because an operation is on torch does not mean it is on GPU. Make sure that all new tensors upon creation are on the same GPU.

# TODO: (Mid priority) which logger are we using and are we using wandb

# NOTE this wrapper is a mix of RslRlVecEnvWrapper and TensorWrapper

# TODO: (Low priority) Remove VecEnv. Actually it is better to have no super class. Make sure all of VecEnv functions transfered correctly to this class
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

        single_action_space = gym.spaces.Box(low=-1, high=1, shape=(self.env.action_space.shape[-1],))
        self.batch_action_space = gym.vector.utils.batch_space(single_action_space, self.num_envs)

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

        self.single_robot_low = torch.from_numpy(self.env.env.single_robot_low).to(self.device).to(torch.float32)
        self.single_robot_high = torch.from_numpy(self.env.env.single_robot_high).to(self.device).to(torch.float32)
        
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
        return self.batch_action_space
    

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}

    def rand_act(self):
        # NOTE: IsaacLab env expects a torch tensor as action input
        # NOTE: IsaacLab runs on 32 bit single precision so env.step  expects a float32 torch tensor
        actions = torch.from_numpy(self.action_space.sample())
        return self._try_f32_tensor(actions)
	
    def _try_f32_tensor(self, x: torch.Tensor):
        if x.dtype == torch.float64:
            x = x.float()
        return x

    def _obs_to_tensor(self, obs: torch.Tensor | dict):
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
        # TODO (Low priority): check if the line below is needed
        assert actions.dtype == torch.float32
        actions = actions.to(self.device)

        action_bound_diff = (self.single_robot_high - self.single_robot_low).unsqueeze(0)
        action_bound_mean = ((self.single_robot_high + self.single_robot_low)/2).unsqueeze(0)
        actions = ((action_bound_diff/2)*actions)+action_bound_mean

        # NOTE: ./rsl_rl/vecenv_wrapper.py enters a torch tensor in the step function 
        obs_dict, rew, terminated, truncated, extras = self.env.step(action=actions)
        # NOTE: You do not need to borrow the info variable because it is only used to calculate success. There is no episode success in this env.
        # NOTE: do not copy the line 'if not self.unwrapped.cfg.is_finite_horizon:' from IsaacLab's vecenv_wrapper because we need access to both pieces of info for now. this might change

        obs = self._obs_to_tensor(obs_dict["policy"])
        rew = self._obs_to_tensor(rew)
        terminated = self._obs_to_tensor(terminated)
        truncated = self._obs_to_tensor(truncated)

        return obs, rew, terminated, truncated, extras

    def max_episode_length(self):
        return self.env.max_episode_length