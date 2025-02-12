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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import quat_rotate
from rsl_rl.datasets.motion_loader import AMPLoader

NO_RAND = True

def play(args):
    register_tasks(args.task, args.seed, NO_RAND=NO_RAND)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.get_commands_from_joystick = False
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False

    train_cfg.runner.amp_num_preload_transitions = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset(random_time=False)
    obs = env.get_observations()

    for repeat_n in range(100):
        env.reset()
        # env.update()
        
        env_ids = torch.arange(env.num_envs, device=env.device)
        while (env.times <= env.amp_loader.trajectory_lens[0] - env.dt):
            frames = env.amp_loader.get_full_frame_at_time_batch(env.traj_idxs, env.times)
            
            env._reset_dofs_amp(env_ids, frames)
            env._reset_root_states_amp(env_ids, frames)
            
            env.times += env.dt
            
            env.update()
            for _ in range(20):
                env.render()
            
    env.reset()
    frames = env.amp_loader.get_full_frame_at_time_batch(env.traj_idxs, env.times)
    temp = frames[0, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]
    frames[0, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX][0]= temp[0] + 1
    env._reset_dofs_amp(env_ids, frames)
    env.update()


if __name__ == '__main__':
    args = get_args()
    # args.task = "al_STMR_AMP"
    play(args)
