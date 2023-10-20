import time

import lcm
import numpy as np
import torch
import cv2

from go1_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import RCControllerProfile
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class LCMAgent():
    def __init__(self, se:StateEstimator, command_profile:RCControllerProfile):
        self.se = se
        self.command_profile = command_profile


        # TODO: Double Check these values
        self.home_joint_pos = np.array([
            0,-0.9,1.8,
            0,-0.9,1.8,
            0,-0.9,1.8,
            0,-0.9,1.8
            ])
        
        # class cfg:
        #     class env:
        #         num_observation_history = 
        
        
    def get_obs(self):
        self.gravity_vector = self.se.get_gravity_vector()
        
        RUN, reset_timer = self.command_profile.get_command(self.timestep * self.dt)
        if reset_timer:
            self.start_time = time.time()

        self.dof_pos = self.se.get_dof_pos()
        self.dof_vel = self.se.get_dof_vel()
        # self.body_linear_vel = self.se.get_body_linear_vel()
        # self.body_angular_vel = self.se.get_body_angular_vel()
        
        self.default_dof_pos = self.motion_loader.get_motion(self.time)

        # TODO: Put OBS here
        obs = torch.cat((   
            self.gravity_vector,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            torch.tensor(self.times, device=self.device, dtype=torch.float32).reshape(-1,1),
            ),dim=-1)

        return torch.tensor(obs, device=self.device).float()

    def get_privileged_observations(self):
        return None

    def publish_action(self, action, hard_reset=False):

        command_for_robot = pd_tau_targets_lcmt()
        self.joint_pos_target = \
            (action[0, :12].detach().cpu().numpy() * self.cfg["control"]["action_scale"]).flatten()
        self.joint_pos_target[[0, 3, 6, 9]] *= self.cfg["control"]["hip_scale_reduction"]
        # self.joint_pos_target[[0, 3, 6, 9]] *= -1
        self.joint_pos_target = self.joint_pos_target
        self.joint_pos_target += self.default_dof_pos
        joint_pos_target = self.joint_pos_target[self.joint_idxs]
        self.joint_vel_target = np.zeros(12)
        # print(f'cjp {self.joint_pos_target}')

        command_for_robot.q_des = joint_pos_target
        command_for_robot.qd_des = self.joint_vel_target
        command_for_robot.kp = self.p_gains
        command_for_robot.kd = self.d_gains
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0

        if hard_reset:
            command_for_robot.id = -1


        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (self.joint_vel_target - self.dof_vel) * self.d_gains

        lc.publish("pd_plustau_targets", command_for_robot.encode())

    def reset(self):
        self.actions = torch.zeros(12)
        self.start_time = time.time()
        self.timestep = 0
        return self.get_obs()

    def step(self, actions, hard_reset=False):
        clip_actions = self.cfg["normalization"]["clip_actions"]
        self.last_actions = self.actions[:]
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        self.publish_action(self.actions, hard_reset=hard_reset)
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0: print(f'frq: {1 / (time.time() - self.time)} Hz');
        self.time = time.time()
        obs = self.get_obs()

        # clock accounting
        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        if self.num_commands == 8:
            bounds = 0
            durations = self.commands[:, 7]
        else:
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        if "pacing_offset" in self.cfg["commands"] and self.cfg["commands"]["pacing_offset"]:
            self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                 self.gait_indices + bounds,
                                 self.gait_indices + offsets,
                                 self.gait_indices + phases]
        else:
            self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                 self.gait_indices + offsets,
                                 self.gait_indices + bounds,
                                 self.gait_indices + phases]
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * self.foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * self.foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * self.foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * self.foot_indices[3])


        images = {'front': self.se.get_camera_front(),
                  'bottom': self.se.get_camera_bottom(),
                  'rear': self.se.get_camera_rear(),
                  'left': self.se.get_camera_left(),
                  'right': self.se.get_camera_right()
                  }
        downscale_factor = 2
        temporal_downscale = 3

        for k, v in images.items():
            if images[k] is not None:
                images[k] = cv2.resize(images[k], dsize=(images[k].shape[0]//downscale_factor, images[k].shape[1]//downscale_factor), interpolation=cv2.INTER_CUBIC)
            if self.timestep % temporal_downscale != 0:
                images[k] = None
        #print(self.commands)

        infos = {"joint_pos": self.dof_pos[np.newaxis, :],
                 "joint_vel": self.dof_vel[np.newaxis, :],
                 "joint_pos_target": self.joint_pos_target[np.newaxis, :],
                 "joint_vel_target": self.joint_vel_target[np.newaxis, :],
                 "body_linear_vel": self.body_linear_vel[np.newaxis, :],
                 "body_angular_vel": self.body_angular_vel[np.newaxis, :],
                 "contact_state": self.contact_state[np.newaxis, :],
                 "clock_inputs": self.clock_inputs[np.newaxis, :],
                 "body_linear_vel_cmd": self.commands[:, 0:2],
                 "body_angular_vel_cmd": self.commands[:, 2:],
                 "privileged_obs": None,
                 "camera_image_front": images['front'],
                 "camera_image_bottom": images['bottom'],
                 "camera_image_rear": images['rear'],
                 "camera_image_left": images['left'],
                 "camera_image_right": images['right'],
                 }

        self.timestep += 1
        return obs, None, None, infos
