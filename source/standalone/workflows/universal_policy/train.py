# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# TODO: make sure the stuff below is correct:
from tdmpc2.trainer import isaaclab_online_trainer
from tdmpc2.common.logger import Logger


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
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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


# TODO: understand what the lines below are and then import them. Also make sure to check why their values are T or F
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = False

# TODO: typecast the agent_cfg below
@hydra_task_config(args_cli.task, "universal_policy_tdmpc2")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
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

    # TODO: make sure you understand the line below or at least run it so you know it 
    # prints obs with env in the batch dimension

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    print("args_cli.task: ", args_cli.task)
    
    # parse config
    # TODO: agent_cfg is in the old agent specific config that was meant to be deleted. We can't have two configs. Reconsile this. cfg = UniversalPolicyTdmpc2() may or may not be the right way to do this. How does the other train.py for rsl_rl do this?
    cfg = UniversalPolicyTdmpc2()
    # TODO: (Low priority) All of the cfg modification operations below should probably replaced with parser like TDMPC does
    cfg.work_dir = os.path.join(cfg.work_dir, 'logs', cfg.task, str(cfg.seed), cfg.exp_name)
    # TODO: (Low priority) this is probably not the best way to get the shape. Change it later.
    sample_obs = env.observation_space.sample()['policy']
    cfg.obs_shape = {"state":sample_obs.shape}
    print(f"cfg.obs_shape {cfg.obs_shape}")
    # TODO: (Low priority) This is redundant. If you deal with the agent_cfg issue the line below isn't required
    cfg.task = args_cli.task
    cfg.task_title = cfg.task.replace("-", " ").title()
    print("cfg.action_dim: ", env.action_space.sample().shape)
    cfg.action_dim = env.action_space.sample().shape

    env = UniversalPolicyWrapper(env)
    trainer = isaaclab_online_trainer.OnlineTrainer(cfg=cfg, env=env, agent=None, buffer=Buffer(cfg), logger=Logger(cfg))
    trainer.train()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()