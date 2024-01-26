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
SIM = True

import glob
from legged_gym import LEGGED_GYM_ROOT_DIR


from legged_gym.envs.go1.go1_config_Common import Go1_Cfg, Go1_CfgPPO, Go1_CfgAMPPPO
from legged_gym.envs.a1.a1_config_Common import A1_Cfg, A1_CfgPPO, A1_CfgAMPPPO
from legged_gym.envs.al.al_config_Common import Al_Cfg, Al_CfgPPO, Al_CfgAMPPPO
from legged_gym.envs.go1base.go1base_config_Common import Go1base_Cfg_PPO, Go1base_Cfg_AMP, Go1base_runner_CfgPPO, Go1base_runner_CfgAMP
from legged_gym.envs.a1base.a1base_config_Common import A1base_Cfg, A1base_CfgPPO, A1base_CfgAMPPPO

ppo_cfg_dict = {
    "go1": (Go1_Cfg, Go1_CfgPPO),
    "a1": (A1_Cfg, A1_CfgPPO),
    "al": (Al_Cfg, Al_CfgPPO),
    "go1base": (Go1base_Cfg_PPO, Go1base_runner_CfgPPO),
    "a1base": (A1base_Cfg, A1base_CfgPPO),
}

amp_cfg_dict = {
    "go1": (Go1_Cfg, Go1_CfgAMPPPO),
    "a1": (A1_Cfg, A1_CfgAMPPPO),
    "al": (Al_Cfg, Al_CfgAMPPPO),
    "go1base": (Go1base_Cfg_AMP, Go1base_runner_CfgAMP),
    "a1base": (A1base_Cfg, A1base_CfgAMPPPO),
}

def get_cfg(ROBOT, MOTION, MR):
    if MR in ["NMR", "SMR", "TMR", "STMR"]:
        common_cfg, common_cfgppo = ppo_cfg_dict[ROBOT.lower()]
    elif MR in ["AMP", "AMPNO", "AMPNONO"]: 
        common_cfg, common_cfgppo = amp_cfg_dict[ROBOT.lower()]
    else:
        raise ValueError(f"MR {MR} not supported")    

    if "base" in ROBOT:
        raw_robot_name = ROBOT.split("base")[0]
    else:
        raw_robot_name = ROBOT
    class Cfg(common_cfg):
        class env(common_cfg.env):
            amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{raw_robot_name}/{MR}/{MOTION}_{raw_robot_name}_{MR}_processed/*')
            total_ee_names = [
                "FL_hip", "FL_thigh", "FL_foot",
                "FR_hip", "FR_thigh", "FR_foot",
                "RL_hip", "RL_thigh", "RL_foot",
                "RR_hip", "RR_thigh", "RR_foot",
                ]
        class rewards(common_cfg.rewards):
            class scales(common_cfg.rewards.scales):
                dof_pos_motion = 3
                pos_motion     = 3
                ang_motion     = 3
                EE_motion      = 30
    
                dof_vel_motion = dof_pos_motion* 0
                lin_vel_motion = pos_motion    * 0
                ang_vel_motion = ang_motion    * 0

    class CfgPPO( common_cfgppo ):
        class runner( common_cfgppo.runner ):
            experiment_name = f"STMR/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}"
            amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{raw_robot_name}/{MR}/{MOTION}_{raw_robot_name}_{MR}_processed/*')

    if MR == 'AMP':
        MR = "NMR"
        Cfg.env.amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{raw_robot_name}/{MR}/{MOTION}_{raw_robot_name}_{MR}_processed/*')
        CfgPPO.runner.amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{raw_robot_name}/{MR}/{MOTION}_{raw_robot_name}_{MR}_processed/*')
    elif MR in ["AMPNO", "AMPNONO"]:
        Cfg.env.amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{raw_robot_name}/NMR/{MOTION}_{raw_robot_name}_NMR_processed/*')
        CfgPPO.runner.amp_motion_files = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{raw_robot_name}/NMR/{MOTION}_{raw_robot_name}_NMR_processed/*')
        
        Cfg.rewards.scales.dof_pos_motion = 0
        Cfg.rewards.scales.pos_motion     = 0
        Cfg.rewards.scales.ang_motion     = 0

        Cfg.rewards.scales.dof_vel_motion = 0
        Cfg.rewards.scales.lin_vel_motion = 0
        Cfg.rewards.scales.ang_vel_motion = 0
    elif MR in ["NMR"]:
        Cfg.env.reference_state_initialization = True
    Cfg.MR = MR

    if SIM:
        Cfg.terrain.curriculum = False
        Cfg.noise.add_noise = False

        Cfg.domain_rand.randomize_gains = False
        Cfg.domain_rand.randomize_base_mass = False
        Cfg.domain_rand.randomize_friction = False
        Cfg.domain_rand.randomize_restitution = False
        Cfg.domain_rand.push_robots = False
        Cfg.domain_rand.randomize_com_displacement = False
        print("SIMULATION MODE: No domain randomization \n" * 50)
    return Cfg, CfgPPO

import os
from legged_gym.utils.task_registry import task_registry

def register_tasks(task, seed):
    ROBOT  = task.split('_')[0].lower()
    MR     = task.split('_')[1].upper()
    MOTION = task.split('_')[2].lower()
    
    register_name = f"{ROBOT.lower()}_{MR}_{MOTION}"

    if MR in ["NMR", "SMR", "TMR", "STMR", "AMP", "AMPNO", "AMPNONO"]:
        Cfg, CfgPPO = get_cfg(ROBOT, MOTION, MR)
    else:
        raise ValueError(f"MR {MR} not supported")
    
    CfgPPO.runner.experiment_name += f"/seed{seed}"
    task_registry.register(register_name, LeggedRobot, Cfg, CfgPPO)
