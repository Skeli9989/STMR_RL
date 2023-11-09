# %%
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
import time
import json
import sys
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

from go1_gym_deploy import BASEDIR
print(BASEDIR.absolute())

ROBOT = 'go1'.lower()
MR = "STMR"
MOTION = "backpace0"

MOTION_FILE = BASEDIR / f"run/{MOTION}/{MR}/{MOTION}_{ROBOT}_{MR}.txt"
POLICY_FILE = BASEDIR / f"run/{MOTION}/{MR}/policy_1.pt"


def load_policy(logdir):
    policy = torch.jit.load(logdir)
    return policy

def load_and_run_policy():
    agent = Go1HardwareAgent()
    
    if POLICY_FILE.exists():
        policy = load_policy(POLICY_FILE)
    else:
        print("There is no policy file!!!")
        sys.exit(0)
        
    if MOTION_FILE.exists():
        motion_file = str(MOTION_FILE)
    else:
        print("There is no motion file!!!")
        sys.exit(0)
    
    deployment_runner = Go1Deployment(agent, policy, motion_file)
    deployment_runner.run()




# %%
# agent = Go1HardwareAgent()
# policy = load_policy(POLICY_FILE)
# motion_file = str(MOTION_FILE)
# deployment_runner = Go1Deployment(agent, policy, motion_file)
# q_array = agent.default_dof_pos.copy()
# # q_array[0] += 0.5
# # # q_array[3] = 0.3
# deployment_runner.smooth_move(q_array)


# agent.reset()
# agent.compute_obs()
# agent.se.joint_pos
# %%

if __name__ == '__main__':
    load_and_run_policy()
