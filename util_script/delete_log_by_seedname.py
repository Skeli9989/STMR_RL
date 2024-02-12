# %%
from pathlib import Path
import glob
import shutil
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

SEED_NAME_TO_DELETE = "seed0"

logs_names = ["logs50", "logs63", "logs64", "logs65", 'logs']

for logs_name in logs_names:
    logs_dir_path = Path(f"{LEGGED_GYM_ROOT_DIR}/{logs_name}")

    for RAND_NORAND in ["RAND", "STMR"]:
        RAND_NORAND_path = logs_dir_path/RAND_NORAND
        if not RAND_NORAND_path.exists(): continue
        for motion_path in (logs_dir_path/RAND_NORAND).iterdir():
            for robot_path in motion_path.iterdir():
                for mr_path in robot_path.iterdir():
                    robot = robot_path.name
                    motion = motion_path.name
                    mr = mr_path.name
                    
                    last_path = mr_path/f"{motion}_{robot}_{mr}"
                    for seed_path in last_path.iterdir():
                        if seed_path.name == SEED_NAME_TO_DELETE:
                            print('deleting', seed_path)
                            shutil.rmtree(seed_path)
                            
from util_script.checklist import main
main()
