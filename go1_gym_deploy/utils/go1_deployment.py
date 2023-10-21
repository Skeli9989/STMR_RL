from go1_gym_deploy.utils.go1_agent import Go1HardwareAgent

import numpy as np
import copy
import torch
import time
import os

class MotionHolder():
    def __init__(self, motion_src):
        self.motion_src = motion_src
        self.q_src = np.array(self.motion_src["Frames"])[:,3+4:3+4+12]
        self.dt = self.motion_src["FrameDuration"]
        self.time_array = np.arange(len(self.q_src)) * self.dt
    
        self.max_time = self.time_array[-1]
        
    def get_q(self, time_):
        idx = np.searchsorted(self.time_array, time_)
        
        if idx == 0:
            print(0)
            return self.q_src[0]
        elif idx == len(self.time_array):
            raise ValueError("time_ is out of range")
        else:
            idx_fr = idx-1
            t1 = self.time_array[idx_fr]
            t2 = self.time_array[idx]
            q1 = self.q_src[idx_fr]
            q2 = self.q_src[idx]
            alpha = (time_ - t1) / (t2 - t1)
            return (1 - alpha) * q1 + alpha * q2
        

class Go1Deployment:
    def __init__(self, agent:Go1HardwareAgent, policy, motion_src):
        self.agent = agent
        self.policy = policy
        self.motion_holder = MotionHolder(motion_src)
    
    def emergeny_stop(self):
        stop_dof_pos = np.array([
                0.0, 1.3, -3.0,
                0.0, 1.3, -3.0,
                0.0, 1.3, -3.0,
                0.0, 1.3, -3.0,                
            ])
        self.smooth_move(stop_dof_pos, duration=1.0)
    
    def smooth_move(self, joint_pos_tar, joint_vel_tar=np.zeros(12), duration=3.0):
        joint_pos = self.agent.se.get_dof_pos()

        dq_size = np.linalg.norm(joint_pos - joint_pos_tar)
        loop_number = int(max(duration/0.05, dq_size/0.05))
        target_squence = np.linspace(joint_pos, joint_pos_tar, loop_number)        
        
        for joint_pos_tar in target_squence:
            joint_vel_tar = np.zeros(12)
            self.agent.publish_joint_target_(joint_pos_tar, joint_vel_tar)
            self.agent.get_obs()
            time.sleep(duration/loop_number)
    
    def calibrate(self, wait=True, low=False):
        agent = self.agent
        agent.reset()
        agent.get_obs()
        
        if low:
            nominal_dof_pos = np.array([
                0.0, 1.4, -2.5,
                0.0, 1.4, -2.5,
                0.0, 1.4, -2.5,
                0.0, 1.4, -2.5,                
            ])

        else:
            nominal_dof_pos = agent.default_dof_pos
        
        print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
        if wait:
            while True:
                if self.agent.se.right_lower_right_switch_pressed:
                    self.agent.se.right_lower_right_switch_pressed = False
                    break

        self.smooth_move(nominal_dof_pos, duration=3.0)

        print("Starting pose calibrated [Press R2 to start controller]")
        while True:
            if self.agent.se.right_lower_right_switch_pressed:
                self.agent.se.right_lower_right_switch_pressed = False
                break
        
        obs = self.agent.reset()
        print("Starting controller !!!")
        return obs
    
    def run(self):
        obs = self.calibrate(wait=True, low=False)
        self.agent.reset()
        
        while self.agent.get_time() < self.motion_holder.max_time - 0.002:
            try:
                obs = self.agent.get_obs()
                # print(obs)
                # breakpoint()
                obs = torch.tensor(obs).to(torch.float).to(device=self.agent.device)
                action = self.policy(obs)
                motion_q = self.motion_holder.get_q(self.agent.get_time())
                self.agent.step(action, motion_q)
                
                rpy = self.agent.se.get_rpy()
                if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                    self.emergeny_stop()
                
            except Exception as e:
                print(e)
                self.emergeny_stop()
                return
            


# %%
