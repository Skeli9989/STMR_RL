# %%
from pathlib import Path
import glob
import shutil
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

ROBOT_NAME_TO_DELETE = ["go1base"]
MOTION_NAME_TO_DELETE = ["trot0", "pace0", "trot1", "pace1"]
MR_NAME_TO_DELETE = ["STMR"]
logs_names = ["logs50", "logs63", "logs64", "logs65", 'logs']

for logs_name in logs_names:
    logs_dir_path = Path(f"{LEGGED_GYM_ROOT_DIR}/{logs_name}")
    if not logs_dir_path.exists(): continue

    for RAND_NORAND in ["RAND", "STMR"]:
        RAND_NORAND_path = logs_dir_path/RAND_NORAND
        if not RAND_NORAND_path.exists(): continue

        for MOTION_NAME in MOTION_NAME_TO_DELETE:
            motion_path = RAND_NORAND_path/MOTION_NAME
            if not motion_path.exists(): continue
            for robot_path in motion_path.iterdir():
                if robot_path.name in ROBOT_NAME_TO_DELETE:
                    for mr_path in robot_path.iterdir():
                        if mr_path.name in MR_NAME_TO_DELETE:
                            print('deteleting', mr_path)
                            shutil.rmtree(robot_path)


from util_script.checklist import main
main()
