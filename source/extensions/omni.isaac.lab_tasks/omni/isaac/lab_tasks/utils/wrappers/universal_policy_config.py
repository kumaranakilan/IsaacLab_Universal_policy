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
    # runtime not by flags but by other stuff in the program
    seed: int = 1
    experiment_name: str = "tdmpc2_a1"
    run_name: str = ""
    resume: bool = False # TODO (low priority): Add this functionality later

    # TMPDC2 examples
    # -------------------------------
    # -------------------------------
    task = 'dog-run'
    obs = 'state'

    # evaluation
    # checkpoint: ??? TODO (low priority): add this in later
    eval_episodes = 10 # 
    eval_freq = 50000 # This frequency will be further divided by the number of envs in train.py for now and later in the cfg parser to be written.

    # training
    single_env_steps = 6_000_000 # NOTE: This should override steps when not not. Calc: steps = single_env_steps/num_envs. This code is now in the train.py and will eventually be moved to the parser.
    steps = 1_000_000 # This variable is used by tdmpc2's code. single_env_steps is used to calculate steps
    num_updates = 1 # Default number of updates when there is just one environment
    batch_size = 256 # training batch size
    reward_coef = 0.1 # the reward_coef is used in weighing the loss
    value_coef = 0.1 # The value_coef is used in weighing the loss
    terminated_coef = 0.1 # termination loss coefficient (float)
    consistency_coef = 20 # The consistency_coef is used in weighing the loss
    rho = 0.5 # rho is the time component weight used in the Q loss
    lr = 3e-4 #  Learning rate
    enc_lr_scale = 0.3 # The lr for the encoder is multiplied by enc_lr_scale
    grad_clip_norm = 20 # This clipping is used to prevent unstable gradients in BPTT
    tau = 0.01 # TODO: (Mid priority) Find out what this is
    discount_denom = 5 # TODO: (Low priority) Find out what this is
    discount_min = 0.95 # TODO: (Low priority) Find out what this is
    discount_max = 0.995 # TODO: (Low priority) Find out what this is
    buffer_size = 1_000_000 # Number of state action pairs the can be stored by the buffer
    exp_name = 'default' # TODO: (Low priority) Find out what this is
    device='cuda' # Obvious

    # planning
    mpc = True # using MPC or just 1 step inference
    iterations = 6 # MPPI iterations # TODO: (Low priority) ONLY as last resort change this to save memory
    num_samples = 512 # number of MPPI samples
    num_elites = 64 # number of MPPI samples considered to average over the next 
    num_pi_trajs = 24 # number of trajectories from NN policy out of the num_samples during MPPI
    horizon = 3 # TODO: (Low priority) play with this term
    min_std = 0.05 # MPPI std min clamp
    max_std = 2 # MPPI std max clamp
    temperature = 0.5 # MMPI average is not a simple average but a softmax weighted average. The temperature param is the sharpness of the weighting within the exponential

    # actor
    log_std_min = -10 # lower limit for policy log std
    log_std_max = 2 # upper limit for policy log std
    entropy_coef = 1e-4 # exploration entropy regularizer

    # critic
    num_bins = 101 # The number of bins used by the critic to predict a discrete distribution over rewards
    vmin = -10 # Min critic bin value
    vmax = +10 # Max critic bin value

    # architecture
    # model_size: ???
    num_enc_layers = 2 # Obvious
    enc_dim = 256 # Obvious
    num_channels = 32 # Obvious
    mlp_dim = 512 # Obvious
    latent_dim = 512 # Obvious
    task_dim = 96 # Obvious
    num_q = 5 # Obvious
    dropout = 0.01 # Obvious
    simnorm_dim = 8 # Obvious

    # logging
    wandb_project = 'Universal_policy_IsaacLab' # Obvious
    wandb_name = 'debug' # Obvious
    wandb_entity = 'deepan_lab' # Obvious
    wandb_silent = True # Obvious
    disable_wandb = False # Obvious
    save_csv = True # Obvious

    # misc
    save_video = False # Obvious
    save_agent = True # Obvious

    # convenience
    work_dir = '~/Documents/Kumaran/universal_policy_isaaclab_logs'
    task_title = "" # Obvious
    multitask = False
    # tasks: ???
    obs_shape = 0 # TODO: (Low priority) this is not a correct default value obs_shape. It should be a shape.
    action_dim = 0 # TODO: (Low priority) this is not a correct default value
    episode_length = 0 # TODO: (Low priority) this is not a correct default value
    # obs_shapes: ???
    action_dims = 0 # TODO: (Low priority) this is not a correct default value
    num_envs = 1 # TODO: (Low priority) How is this routed again???
    # episode_lengths: ???
    seed_steps = 100 
    bin_size = 0.2 # TODO: (Low priority) Find out what this is