# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# TODO: (Low priority) delete this file

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from tdmpc2.common.parser import parse_cfg

@configclass
class UniversalPolicyTdmpc2():
    # NOTE: I removed all of the variables here because we will be deleting this. Likley but not sure
    pass