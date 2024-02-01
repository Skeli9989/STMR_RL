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

# %%

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from isaacgym import gymtorch, gymapi, gymutil

import pickle 
from pathlib import Path
from argparse import Namespace
import onnxruntime as ort
import numpy as np

# %%


def export_onnx(task, seed=1, NO_RAND=False, device = 'cpu'):
    with open(LEGGED_GYM_ROOT_DIR/Path('datasets/default.pkl'), 'rb') as f:
        args = pickle.load(f)

    args = Namespace(**args)
    args.task = task
    args.seed = seed

    ROBOT = args.task.split('_')[0]
    MR = args.task.split('_')[1]
    MOTION = args.task.split('_')[2]

    register_tasks(args.task, args.seed, NO_RAND=NO_RAND)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset(random_time=False)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)

    device = 'cpu'
    torch_model = ppo_runner.alg.actor_critic.to(device=device)
    torch_model.forward = torch_model.act_inference
    torch_input = torch.ones(1,env_cfg.env.num_observations).to(device=device)

    torch_model(torch_input)

    if 'base' in ROBOT:
        raw_robot_name = ROBOT.split("base")[0]
    else:
        raw_robot_name = ROBOT

    savename = f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{raw_robot_name}/{MR}/{MOTION}_{ROBOT}_{MR}.onnx'
    onnx_program = torch.onnx.export(
        torch_model, 
        torch_input,
        savename,
        opset_version=9,
        input_names=["input"],
        output_names=["action"])

    ort_sess = ort.InferenceSession(savename)
    outputs = ort_sess.run(None, {'input': torch_input.numpy()})

    print("difference", outputs - torch_model(torch_input).detach().numpy())
