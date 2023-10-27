from go1_gym_deploy.utils.go1_agent import Go1HardwareAgent
from go1_gym_deploy.utils.motion_holder import MotionHolder

import numpy as np
import copy
import torch
import time
import os

        

class Go1Deployment:
    def __init__(self, agent:Go1HardwareAgent, policy, motion_file):
        self.agent = agent
        self.policy = policy
        self.motion_holder = MotionHolder(motion_file)
    
    def emergeny_stop(self):
        stop_dof_pos = np.array([
                0.0, 1.3, -3.0,
                0.0, 1.3, -3.0,
                0.0, 1.3, -3.0,
                0.0, 1.3, -3.0,                
            ])
        self.smooth_move(stop_dof_pos, duration=1.0, d_gains=10, p_gains = 40)
    
    def smooth_move(self, joint_pos_tar, joint_vel_tar=np.zeros(12), duration=3.0, p_gains=None, d_gains=None):
        joint_pos = self.agent.se.get_dof_pos()

        dq_size = np.linalg.norm(joint_pos - joint_pos_tar)
        loop_number = int(max(duration/0.05, dq_size/0.05))
        target_squence = np.linspace(joint_pos, joint_pos_tar, loop_number)        
        
        for joint_pos_tar in target_squence:
            joint_vel_tar = np.zeros(12)
            self.agent.publish_joint_target_(joint_pos_tar, joint_vel_tar, p_gains=p_gains, d_gains=d_gains)
            self.agent.get_obs()
            time.sleep(duration/loop_number)
    
    def calibrate(self, wait=True, low=False, nominal_dof_pos =None) :
        agent = self.agent
        agent.reset()
        agent.get_obs()
        
        if nominal_dof_pos is None:
            if low:
                nominal_dof_pos = np.array([
                    0.0, 1.4, -2.5,
                    0.0, 1.4, -2.5,
                    0.0, 1.4, -2.5,
                    0.0, 1.4, -2.5,                
                ])

            else:
                nominal_dof_pos = agent.default_dof_pos
        else:
            nominal_dof_pos = nominal_dof_pos
            
        print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
        if wait:
            while True:
                if self.agent.se.right_lower_right_switch_pressed:
                    self.agent.se.right_lower_right_switch_pressed = False
                    break

        self.smooth_move(nominal_dof_pos, duration=3.0, p_gains=80, d_gains=1.0)

        obs = self.agent.reset()
        obs_history = self.agent.get_obs()
        print(obs_history)
        print("Starting pose calibrated [Press R2 to start controller]")
        while True:
            if self.agent.se.right_lower_right_switch_pressed:
                self.agent.se.right_lower_right_switch_pressed = False
                break
        
        obs = self.agent.reset()
        print("Starting controller !!!")
        return obs
    
    def run(self):
        action_list = []
        obs_list    = []
        
        motion_q = self.motion_holder.get_q(0)
        self.calibrate(wait=False, low=False, nominal_dof_pos=motion_q)
        self.agent.reset()
        obs_history = self.agent.get_obs()
        obs_history = torch.tensor(obs_history).to(torch.float).to(device=self.agent.device)
        action = self.policy(obs_history)
        
        try:
            self.agent.reset()
            motion_q = self.motion_holder.get_q(self.agent.get_time())
            self.agent.step(action, motion_q)
            action_list.append(action)
            obs_list.append(self.agent.obs)
        except Exception as e:
            print(e)
            self.emergeny_stop()
            return
        
        while self.agent.get_time() < self.motion_holder.max_time:
            try:
                obs_history = self.agent.get_obs()
                # print(obs)
                # breakpoint()
                obs_history = torch.tensor(obs_history).to(torch.float).to(device=self.agent.device)
                action = self.policy(obs_history)
                action_list.append(action)
                print(self.agent.obs)
                obs_list.append(self.agent.obs)
                motion_q = self.motion_holder.get_q(self.agent.get_time())
                self.agent.step(action, motion_q)
                
                rpy = self.agent.se.get_rpy()
                if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                    self.emergeny_stop()
                
            except Exception as e:
                print(e)
                self.emergeny_stop()
                return

        action_list = torch.stack(action_list)
        obs_list = np.stack(obs_list)
        # save to txt from torch tensor
        np.savetxt("action_list.txt", action_list.detach().cpu().numpy())
        np.savetxt("obs_list.txt", obs_list)

        while True:
            try:
                self.agent.step(action, motion_q)
            except Exception as e:
                print(e)
                self.emergeny_stop()
                return

# %%
