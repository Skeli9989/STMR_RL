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
from rsl_rl.datasets.motion_loader import AMPLoader

def play(args):
    register_tasks(args.task)

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

    env.reset(random_time=False)
    
    if args.load_run is None:
        root= f"{LEGGED_GYM_ROOT_DIR}/logs/{train_cfg.runner.experiment_name}"
        runs = os.listdir(root)
        if 'exported' in runs: runs.remove('exported')
        runs.sort()
        load_run = os.path.join(root, runs[-1])
    else:
        load_run = f"{LEGGED_GYM_ROOT_DIR}/logs/{train_cfg.runner.experiment_name}/{args.load_run}"
    
    from pathlib import Path
    GET_ALL = False
    if GET_ALL:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        save_path = Path(LEGGED_GYM_ROOT_DIR)/f"performance/{train_cfg.runner.experiment_name}/performance_all.json"
    else:
        models = ["model_10000.pt"]
        save_path = Path(LEGGED_GYM_ROOT_DIR)/f"performance/{train_cfg.runner.experiment_name}/performance_1k.json"

    res_dict = {}
    for model in models:
        iternum = str.split(str.split(model,"_")[1], ".pt")[0]
        train_cfg.runner.checkpoint = int(iternum)
        # load policy
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)

        res_dict_ = dict(
            joint_tracking_error = [],
            position_tracking_error = [],
            orientation_tracking_error = []
        )
        
        env.reset(random_time=False)
        while env.times <= env.amp_loader.trajectory_lens[0] - env.dt:
            actions = policy(obs.detach())
            obs, _, rews, dones, infos, _, _ = env.step(actions.detach(), RESET_ABLED=False)
        
            env.times = np.clip(env.times, 0, env.amp_loader.trajectory_lens[0] - env.amp_loader.trajectory_frame_durations[0])
            
            frames = env.amp_loader.get_full_frame_at_time_batch(env.traj_idxs, env.times)
            frames = frames.to(env.device)
            
            dof_pos = AMPLoader.get_joint_pose_batch(frames)
            dof_pos_error = torch.sum(torch.square(dof_pos - env.dof_pos))
            res_dict_["joint_tracking_error"].append(dof_pos_error.detach().cpu().tolist())
            
            root_pos = AMPLoader.get_root_pos_batch(frames)
            root_pos[:,:2] += env.env_origins[:, :2]
            root_pos_error = torch.sum(torch.square(root_pos - env.root_states[:, 0:3]))
            res_dict_["position_tracking_error"].append(root_pos_error.detach().cpu().tolist())

            root_rot = AMPLoader.get_root_rot_batch(frames)
            root_rot_cur= env.root_states[:,3:7]

            inner_product = torch.sum(root_rot_cur * root_rot)
            ang_error =  1 - inner_product ** 2
            res_dict_["orientation_tracking_error"].append(ang_error.detach().cpu().tolist())

        for key,value in res_dict_.items():
            res_dict_[key] = np.mean(value)
        res_dict[iternum] = res_dict_
        from toolbox.write import write_json
        write_json(save_path, res_dict)

if __name__ == '__main__':
    args = get_args()
    args.task = "go1base_STMR_hopturn"
    # args.task = "a1_amp"
    play(args)
