# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from tdmpc2.common.parser import parse_cfg

# NOTE: The reason the config was moved here is because it is not specific to any env but to all of tdmpc2

@configclass
class UniversalPolicyTdmpc2():
    # TODO: print out on the tdmpc2 side and see which config values are changed during 
    # runtime not by flags but by other stuff in the program
    seed: int = 42
    experiment_name: str = "tdmpc2_a1"
    run_name: str = ""
    resume: bool = False
    max_iterations: int = 300 # TODO: not sure where this came from???

    # TMPDC2 examples
    # -------------------------------
    # -------------------------------
    task = 'dog-run'
    obs = 'state'

    # evaluation
    # checkpoint: ???
    eval_episodes = 10
    eval_freq = 50000

    # training
    steps = 10_000_000
    batch_size = 256
    reward_coef = 0.1
    value_coef = 0.1
    consistency_coef = 20
    rho = 0.5
    lr = 3e-4
    enc_lr_scale = 0.3
    grad_clip_norm = 20
    tau = 0.01
    discount_denom = 5
    discount_min = 0.95
    discount_max = 0.995
    buffer_size = 1_000_000
    exp_name = 'default'
    # data_dir = ???

    # planning
    mpc = True
    iterations = 6
    num_samples = 512
    num_elites = 64
    num_pi_trajs = 24
    horizon = 3
    min_std = 0.05
    max_std = 2
    temperature = 0.5

    # actor
    log_std_min = -10
    log_std_max = 2
    entropy_coef = 1e-4

    # critic
    num_bins = 101
    vmin = -10
    vmax = +10

    # architecture
    # model_size: ???
    num_enc_layers = 2
    enc_dim = 256
    num_channels = 32
    mlp_dim = 512
    latent_dim = 512
    task_dim = 96
    num_q = 5
    dropout = 0.01
    simnorm_dim = 8

    # logging
    wandb_project = 'Universal_policy_IsaacLab'
    wandb_name = 'debug'
    wandb_entity = 'deepan_lab'
    wandb_silent = True
    disable_wandb = False
    save_csv = True

    # misc
    save_video = True
    save_agent = True

    # convenience
    work_dir = '~/Documents/Kumaran/universal_policy_isaaclab_logs'
    task_title = ""
    multitask = False
    # tasks: ???
    obs_shape = 0 # TODO: this is not a correct default value obs_shape. It should be a shape.
    action_dim = 0
    # episode_length: ???
    # obs_shapes: ???
    action_dims = 0
    # episode_lengths: ???
    # seed_steps: ???
    # bin_size: ???