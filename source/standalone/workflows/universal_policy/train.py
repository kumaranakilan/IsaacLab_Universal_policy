# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

from tdmpc2.trainer import isaaclab_online_trainer
from tdmpc2.common.logger import Logger
from tdmpc2.tdmpc2 import TDMPC2


# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime


from tdmpc2.common.buffer import Buffer


from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from omni.isaac.lab_tasks.utils.wrappers import UniversalPolicyWrapper
from omni.isaac.lab_tasks.utils.wrappers import UniversalPolicyTdmpc2


# TODO: (Mid priority) understand what the lines below are and then import them. Also make sure to check why their values are T or F
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = False

# TODO: (Low priority) typecast the agent_cfg below
@hydra_task_config(args_cli.task, "universal_policy_tdmpc2")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    # NOTE: If you want to turn off terminations comment out self.terminations.base_contact. ... in rough_env_cfg.py and base_contact = DoneTerm in velocity_env_cfg.py
    args_cli.num_envs = 8 # TODO: (Low priority) switch to specifiying this through command line
    print("env_cfg.terminations: ", env_cfg.terminations)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # TODO: (Low priority) change the num_envs value as you need better compute
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.swap_reset_order = True

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "universal_policy", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    print("env_cfg.scene.num_envs: ", env_cfg.scene.num_envs)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    print("env_cfg.undesired_contacts: ", env_cfg.rewards.undesired_contacts)
    
    # TODO: (medium priority) Figure out how to add video saving as it is likely needed for debugging
    # parse config
    # TODO: (Low priority) agent_cfg is in the old agent specific config that was meant to be deleted. We can't have two configs. Reconsile this. cfg = UniversalPolicyTdmpc2() may or may not be the right way to do this. How does the other train.py for rsl_rl do this?
    cfg = UniversalPolicyTdmpc2()
    # TODO: (Low priority) All of the cfg modification operations below should probably replaced with parser like TDMPC does
    cfg.work_dir = os.path.join(cfg.work_dir, 'logs', cfg.task, str(cfg.seed), cfg.exp_name)
    # TODO: (Low priority) this is probably not the best way to get the shape. Change it later.
    sample_obs = env.observation_space.sample()['policy']
    cfg.obs_shape = {"state":sample_obs.shape[1:]}
    print(f"cfg.obs_shape {cfg.obs_shape}")
    # TODO: (Low priority) This is redundant. If you deal with the agent_cfg issue the line below isn't required
    cfg.task = args_cli.task
    cfg.task_title = cfg.task.replace("-", " ").title()
    print("cfg.action_dim: ", env.action_space.sample().shape)
    cfg.action_dim = env.action_space.sample().shape[-1]

    # Divide the number of steps for training by the number of envs
    cfg.steps = int(cfg.single_env_steps/env_cfg.scene.num_envs)
    # Multiply the number of updates for training by the number of envs
    cfg.num_updates = cfg.num_updates*env_cfg.scene.num_envs
    # Divide the eval_freq for training by the number of envs
    cfg.eval_freq = cfg.eval_freq/env_cfg.scene.num_envs

    env = UniversalPolicyWrapper(env)
    print("env.max_episode_length: ", env.max_episode_length)
    cfg.episode_length = env.max_episode_length
    cfg.task_dim = 0
    cfg.num_envs = env_cfg.scene.num_envs

    agent = TDMPC2(cfg)
    # TODO: (Medium priority) I don't know if the original tdmpc2 training code disables MPC for the initial iterations or not. If it is make sure the functionality is still on.
    trainer = isaaclab_online_trainer.OnlineTrainer(cfg=cfg, env=env, agent=agent, buffer=Buffer(cfg), logger=Logger(cfg))
    trainer.train()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()