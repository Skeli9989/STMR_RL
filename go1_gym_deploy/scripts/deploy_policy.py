import glob
import pickle as pkl
import lcm
import sys
from go1_gym_deploy.utils.go1_state_estimator import GO1StateEstimator
from go1_gym_deploy.utils.go1_agent import Go1HardwareAgent
from go1_gym_deploy.utils.go1_deployment import Go1Deployment
import pathlib
import torch
import numpy as np
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def temp():
    for _ in range(100):
        action = agent.se.get_dof_pos()
        import time
        agent.publish_joint_target_(action, np.zeros_like(action))
        time.sleep(0.1)



def load_and_run_policy(label, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    agent = Go1HardwareAgent()
    policy = None
    deployment_runner = Go1Deployment(agent, policy)
    deployment_runner.calibrate()

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


if __name__ == '__main__':
    label = "gait-conditioned-agility/pretrain-v0/train"

    experiment_name = "example_experiment"

    load_and_run_policy(label, experiment_name=experiment_name, max_vel=3.5, max_yaw_vel=5.0)
