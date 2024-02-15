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
from toolbox.write import write_json

import mujoco
from mujoco_viewer.mujoco_viewer import MujocoViewer
import numpy as np
from numpy.linalg import norm
from toolbox.resources import get_asset_dict, ASSET_SRC
from mjmr.util import reset, get_mr_info, get_xml_path, plot_contact_schedule, get_vel_contact_boolean, get_mujoco_contact_boolean,get_vel_contact_boolean_from_pos, get_key_id, output_amp_motion
from mjmr.task.Quadruped.info import QuadrupedRetargetInfo as RetargetInfo
from mjpc.task.Quadruped.play import plot_skeleton
from mjpc.task.Quadruped.info import QuadrupedMPCInfo as MPCInfo

from fastdtw import fastdtw
from scipy.spatial.distance import cityblock

NO_RAND = True
GET_ALL = True
RENDER = False

def get_target_deploy_array(env, model, train_cfg, obs):
    iternum = str.split(str.split(model,"_")[1], ".pt")[0]
    train_cfg.runner.checkpoint = int(iternum)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    target_ls = []
    deploy_ls = []
    env.reset(random_time=False)
    while env.times < env.amp_loader.trajectory_lens[0] - env.amp_loader.trajectory_frame_durations[0]:
        actions = policy(obs.detach())
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach(), RESET_ABLED=False)
    
        env.times = np.clip(env.times, 0, env.amp_loader.trajectory_lens[0] - env.amp_loader.trajectory_frame_durations[0])
        
        frames = env.amp_loader.get_full_frame_at_time_batch(env.traj_idxs, env.times)
        frames = frames.to(env.device)
        
        dof_pos = AMPLoader.get_joint_pose_batch(frames)
        
        root_pos = AMPLoader.get_root_pos_batch(frames)
        root_pos[:,:2] += env.env_origins[:, :2]

        root_rot = AMPLoader.get_root_rot_batch(frames)
        root_rot_cur= env.root_states[:,3:7]


        def cast_numpy(tensor):
            return tensor.detach().cpu().numpy().copy().flatten()
        
        target = np.concatenate([cast_numpy(root_pos), cast_numpy(root_rot), cast_numpy(dof_pos)])
        deploy = np.concatenate([cast_numpy(env.root_states[:, 0:3]), cast_numpy(env.root_states[:, 3:7]), cast_numpy(env.dof_pos)])

        # res_dict["target"].append(target.tolist())
        # res_dict["deploy"].append(deploy.tolist())
        target_ls.append(target.tolist())
        deploy_ls.append(deploy.tolist())

    return target_ls, deploy_ls

def play(args):
    ROBOT  = args.task.split("_")[0]
    MR     = args.task.split("_")[1]
    MOTION = args.task.split("_")[2]

    if 'base' in ROBOT:
        ROBOT = ROBOT.split("base")[0]
    ROBOT_base = ROBOT+"base"

    # Load Mujoco Model
    xml_path = get_xml_path(ROBOT)
    mjmodel = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    mjdata  = mujoco.MjData(mjmodel)
    reset(mjmodel, mjdata)

    if RENDER:
        try:
            viewer.close()
        except Exception:
            pass

        viewer = MujocoViewer(
            mjmodel,mjdata,mode='window',title="MPC",
            width=1200,height=800,hide_menus=True
        )
        args.headless = False
    else:
        viewer = None
        args.headless = True
    mr_info   = RetargetInfo(mjmodel, mjdata)
    mpc_info = MPCInfo(mjmodel, mjdata)



    # Load Isaac Model
    register_tasks(args.task, args.seed, NO_RAND=NO_RAND)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1) # override some parameters for testing

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset(random_time=False)
    obs = env.get_observations()

    env.reset(random_time=False)
    env.reset_idx(torch.arange(env.num_envs, device=env.device), random_time=False)
    
    if args.load_run is None:
        root= f"{LEGGED_GYM_ROOT_DIR}/logs/{train_cfg.runner.experiment_name}"
        runs = os.listdir(root)
        if 'exported' in runs: runs.remove('exported')
        runs.sort()
        load_run = os.path.join(root, runs[-1])
    else:
        load_run = f"{LEGGED_GYM_ROOT_DIR}/logs/{train_cfg.runner.experiment_name}/{args.load_run}"
    
    from pathlib import Path

    models_names = [file for file in os.listdir(load_run) if 'model' in file]
    models_names.sort(key=lambda m: '{0:0>15}'.format(m))
    if GET_ALL:
        save_path = Path(LEGGED_GYM_ROOT_DIR)/f"performance/{train_cfg.runner.experiment_name}/pose_all.json"
    else:
        models_names = [models_names[-1]]
        save_path = Path(LEGGED_GYM_ROOT_DIR)/f"performance/{train_cfg.runner.experiment_name}/pose_1k.json"

    res_dict = {"target":[], "deploy":[], "dtw_distance":[]}
    for model_name in models_names:
        print(f"{model_name}_{models_names[-1]}")
        target_ls, deploy_ls = get_target_deploy_array(env, model_name, train_cfg, obs)

        res_dict["target"].append(target_ls)
        res_dict["deploy"].append(deploy_ls)

        dtw_distance = calculate_dtw_distance(mjmodel, mjdata, mr_info, mpc_info, target_ls, deploy_ls, viewer)
        res_dict["dtw_distance"].append(dtw_distance)
        write_json(save_path, res_dict)

def calculate_dtw_distance(mjmodel, mjdata, mr_info, mpc_info, target_ls, deploy_ls,viewer, render_every=10):
    site_ids = [
        mr_info.id.trunk_site,
        mr_info.id.FR_hip_site, mr_info.id.FR_calf_site, mr_info.id.FR_foot_site,
        mr_info.id.FL_hip_site, mr_info.id.FL_calf_site, mr_info.id.FL_foot_site,
        mr_info.id.RR_hip_site, mr_info.id.RR_calf_site, mr_info.id.RR_foot_site,
        mr_info.id.RL_hip_site, mr_info.id.RL_calf_site, mr_info.id.RL_foot_site,
        ]
    mujoco_idx = [3,4,5,6, 7,8,9, 10,11,12, 13,14,15, 16,17,18]
    amp_idx    = [6,3,4,5, 10,11,12, 7,8,9, 16,17,18, 13,14,15]

    target_qpos_total = np.array(target_ls)
    target_qpos_total[:,mujoco_idx] = target_qpos_total[:,amp_idx].copy()

    deploy_qpos_total = np.array(deploy_ls)
    deploy_qpos_total[:,mujoco_idx] = deploy_qpos_total[:,amp_idx].copy()

    target_qpos_total[:, 7:] +=  mjmodel.qpos0[7:] 
    deploy_qpos_total[:, 7:] +=  mjmodel.qpos0[7:]

    frame_number = len(target_qpos_total)

    target_site_ls = []
    deploy_site_ls = []
    for frame_i in range(frame_number):
        target_qpos = target_qpos_total[frame_i]
        deploy_qpos = deploy_qpos_total[frame_i]
        mjdata.qpos = target_qpos.copy()
        mujoco.mj_forward(mjmodel, mjdata)
        for site_id in site_ids:
            target_site_ls.append(mjdata.site_xpos[site_id].copy())
        
        if frame_i%render_every == 0:
            if RENDER: plot_skeleton(mjmodel, mjdata, mpc_info, viewer, rgba=[1,0,0,0.5])

        mjdata.qpos = deploy_qpos.copy()
        mujoco.mj_forward(mjmodel, mjdata)
        for site_id in site_ids:
            deploy_site_ls.append(mjdata.site_xpos[site_id].copy())
        if frame_i%render_every == 0:
            if RENDER: viewer.render()
    
    target_site_array = np.array(target_site_ls).reshape(frame_number, -1)
    deploy_site_array = np.array(deploy_site_ls).reshape(frame_number, -1)

    distance, path = fastdtw(target_site_array, deploy_site_array, dist=cityblock)
    key_point_error = distance/frame_number/len(site_ids)

    return key_point_error

if __name__ == '__main__':
    args = get_args()
    play(args)
    # if GET_ALL:
    #     from util_script.plot_performance import main as plot_main
    #     plot_main()