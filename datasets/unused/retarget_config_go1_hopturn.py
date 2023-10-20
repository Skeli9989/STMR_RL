import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR


VISUALIZE_RETARGETING = True

URDF_FILENAME = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

OUTPUT_DIR = "{LEGGED_GYM_ROOT_DIR}/datasets/hopturn_go1".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

INIT_POS = np.array([0, 0, 0.27])
INIT_ROT = np.array([0, 0, 0, 1.0])

DEFAULT_JOINT_POSE = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])

FR_FOOT_NAME = "FR_foot"
FL_FOOT_NAME = "FL_foot"
HR_FOOT_NAME = "RR_foot"
HL_FOOT_NAME = "RL_foot"

MOCAP_MOTIONS = [
    # Output motion name, input file, frame start, frame end, motion weight.
    [
        "hopturn",
        "{LEGGED_GYM_ROOT_DIR}/datasets/hopturn/hopturn.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 0, 80, 1

    ]
]