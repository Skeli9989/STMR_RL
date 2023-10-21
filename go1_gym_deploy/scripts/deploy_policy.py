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

# def temp():
#     for _ in range(100):
#         action = agent.se.get_dof_pos()
#         import time
#         agent.publish_joint_target_(action, np.zeros_like(action))
#         time.sleep(0.1)

from go1_gym_deploy import BASEDIR
print(BASEDIR.absolute())

ROBOT = 'go1'.lower()
MR = "TMR"
MOTION = "hopturn"

MOTION_FILE = BASEDIR / f"run/{MOTION}/{MR}/{MOTION}_{ROBOT}_{MR}.txt"
POLICY_FILE = BASEDIR / f"run/{MOTION}/{MR}/policy_1.pt"


def load_and_run_policy():
    agent = Go1HardwareAgent()
    
    if POLICY_FILE.exists():
        policy = load_policy(POLICY_FILE)
    else:
        print("There is no policy file!!!")
        sys.exit(0)
        
    if MOTION_FILE.exists():
        motion_src = load_motion_file(str(MOTION_FILE))
    else:
        print("There is no motion file!!!")
        sys.exit(0)
    
    
    deployment_runner = Go1Deployment(agent, policy, motion_src)
    deployment_runner.run()

def load_policy(logdir):
    policy = torch.jit.load(logdir)
    return policy

def load_motion_file(file_path):
    with open(file_path) as f:
        return json.load(f)


# %%





# %%

if __name__ == '__main__':
    load_and_run_policy()
