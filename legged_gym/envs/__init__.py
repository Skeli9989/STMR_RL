# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

# %%
import glob
from legged_gym import LEGGED_GYM_ROOT_DIR


from legged_gym.envs.go1.go1_config_Common import Go1_Cfg
from legged_gym.envs.go1.go1_config_Common import Go1_CfgPPO

from legged_gym.envs.a1.a1_config_Common import A1_Cfg
from legged_gym.envs.a1.a1_config_Common import A1_CfgPPO

from legged_gym.envs.al.al_config_Common import Al_Cfg
from legged_gym.envs.al.al_config_Common import Al_CfgPPO

common_config_dict = {
    "go1": (Go1_Cfg, Go1_CfgPPO),
    "a1": (A1_Cfg, A1_CfgPPO),
    "al": (Al_Cfg, Al_CfgPPO),
}

def get_NMR_cfg(ROBOT, MOTION):
    MR = "NMR"
    common_cfg, common_cfgppo = common_config_dict[ROBOT.lower()]
    
    class Cfg(common_cfg):
        class env(common_cfg.env):
            amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_processed/*')
        
        class rewards(common_cfg.rewards):
            class scales(common_cfg.rewards.scales):
                pos_motion     = 150 * 3
                ang_motion     = 150 * 3

    class CfgPPO( common_cfgppo ):
        class runner( common_cfgppo.runner ):
            experiment_name = f"STMR/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}"
            amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_processed/*')

    return Cfg, CfgPPO


def get_SMR_cfg(ROBOT, MOTION):
    MR = "SMR"
    common_cfg, common_cfgppo = common_config_dict[ROBOT.lower()]
    
    class Cfg(common_cfg):
        class env(common_cfg.env):
            amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_processed/*')
        
        class rewards(common_cfg.rewards):
            class scales(common_cfg.rewards.scales):
                pos_motion     = 150
                ang_motion     = 150
                dof_pos_motion = 150

                dof_vel_motion = 50
                lin_vel_motion = 50
                ang_vel_motion = 50


    class CfgPPO( common_cfgppo ):
        class runner( common_cfgppo.runner ):
            experiment_name = f"STMR/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}"
            amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_processed/*')

    return Cfg, CfgPPO

def get_TMR_cfg(ROBOT,MOTION):
    MR = "TMR"
    common_cfg, common_cfgppo = common_config_dict[ROBOT.lower()]
    
    class Cfg(common_cfg):
        class env(common_cfg.env):
            amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_processed/*')
        
        class rewards(common_cfg.rewards):
            class scales(common_cfg.rewards.scales):
                pos_motion     = 150
                ang_motion     = 150
                dof_pos_motion = 150

                dof_vel_motion = 50
                lin_vel_motion = 50
                ang_vel_motion = 50


    class CfgPPO( common_cfgppo ):
        class runner( common_cfgppo.runner ):
            experiment_name = f"STMR/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}"
            amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_processed/*')

    return Cfg, CfgPPO


def get_STMR_cfg(ROBOT,MOTION):
    MR = "STMR"
    common_cfg, common_cfgppo = common_config_dict[ROBOT.lower()]
    
    class Cfg(common_cfg):
        class env(common_cfg.env):
            amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_processed/*')
        
        class rewards(common_cfg.rewards):
            class scales(common_cfg.rewards.scales):
                pos_motion     = 150/5
                ang_motion     = 150/5
                dof_pos_motion = 150

                dof_vel_motion = 50
                lin_vel_motion = 50/5
                ang_vel_motion = 50/5
                
                # pos_motion     = 30
                # ang_motion     = 30
                # dof_pos_motion = 30

                # dof_vel_motion = 2
                # lin_vel_motion = 2
                # ang_vel_motion = 2


    class CfgPPO( common_cfgppo ):
        class runner( common_cfgppo.runner ):
            experiment_name = f"STMR/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}"
            amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_processed/*')

    return Cfg, CfgPPO


import os
from legged_gym.utils.task_registry import task_registry


def register_tasks(task):
    ROBOT  = task.split('_')[0].lower()
    MR     = task.split('_')[1].upper()
    MOTION = task.split('_')[2].lower()
    
    register_name = f"{ROBOT.lower()}_{MR}_{MOTION}"

    if MR == "NMR":
        Cfg, CfgPPO = get_NMR_cfg(ROBOT, MOTION)
    elif MR == "SMR":
        Cfg, CfgPPO = get_SMR_cfg(ROBOT, MOTION)
    elif MR == "TMR":
        Cfg, CfgPPO = get_TMR_cfg(ROBOT, MOTION)
    elif MR == "STMR":
        Cfg, CfgPPO = get_STMR_cfg(ROBOT, MOTION)
    else:
        print(f"Unknow task: {task}")
        raise NotImplementedError    
    
    task_registry.register(register_name, LeggedRobot, Cfg, CfgPPO)

# task_registry.register("a1_NMR_AMP", LeggedRobot, A1_NMR_Cfg(), A1_NMR_AMP())
# task_registry.register("a1_TMR_AMP", LeggedRobot, A1_TMR_Cfg(), A1_TMR_AMP())
# task_registry.register("a1_SMR_AMP", LeggedRobot, A1_SMR_Cfg(), A1_SMR_AMP())
# task_registry.register("a1_STMR_AMP", LeggedRobot, A1_STMR_Cfg(), A1_STMR_AMP())


# task_registry.register("go1_NMR_AMP", LeggedRobot, GO1_NMR_Cfg(), GO1_NMR_AMP())
# task_registry.register("go1_TMR_AMP", LeggedRobot, GO1_TMR_Cfg(), GO1_TMR_AMP())
# task_registry.register("go1_SMR_AMP", LeggedRobot, GO1_SMR_Cfg(), GO1_SMR_AMP())
# task_registry.register("go1_STMR_AMP", LeggedRobot, GO1_STMR_Cfg(), GO1_STMR_AMP())

# task_registry.register("al_NMR_AMP", LeggedRobot, AL_NMR_Cfg(), AL_NMR_AMP())
# task_registry.register("al_TMR_AMP", LeggedRobot, AL_TMR_Cfg(), AL_TMR_AMP())
# task_registry.register("al_SMR_AMP", LeggedRobot, AL_SMR_Cfg(), AL_SMR_AMP())
# task_registry.register("al_STMR_AMP", LeggedRobot, AL_STMR_Cfg(), AL_STMR_AMP())