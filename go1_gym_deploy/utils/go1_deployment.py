from go1_gym_deploy.utils.go1_agent import Go1HardwareAgent

import numpy as np
import copy
import torch
import time

class Go1Deployment:
    def __init__(self, hardware_agent:Go1HardwareAgent, policy):
        self.hardware_agent = hardware_agent
        self.policy = policy
        
    def calibrate(self, wait=True, low=False):
        agent = self.hardware_agent
        agent.reset()
        agent.get_obs()
        
        joint_pos = agent.dof_pos
        if low:
            nominal_dof_pos = np.array([
                0.0, 0.0, -0.7,
                0.0, 0.0, -0.7,
                0.0, 0.0, -0.7,
                0.0, 0.0, -0.7,                
            ])
        else:
            nominal_dof_pos = agent.default_dof_pos
        
        print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
        if wait:
            while True:
                if self.hardware_agent.se.right_lower_right_switch_pressed:
                    self.hardware_agent.se.right_lower_right_switch_pressed = False
                    break

        dq_size = np.linalg.norm(joint_pos - nominal_dof_pos)
        loop_number = int(max(5/0.01, dq_size/0.05))
        target_squence = np.linspace(joint_pos, nominal_dof_pos, loop_number)        
        
        for joint_pos_tar in target_squence:
            joint_vel_tar = np.zeros(12)
            agent.publish_joint_target_(joint_pos_tar, joint_vel_tar)
            agent.get_obs()
            time.sleep(0.01)

        print("Starting pose calibrated [Press R2 to start controller]")
        while True:
            if self.hardware_agent.se.right_lower_right_switch_pressed:
                self.hardware_agent.se.right_lower_right_switch_pressed = False
                break
        
        obs = self.hardware_agent.reset()
        print("Starting controller !!!")
        return obs