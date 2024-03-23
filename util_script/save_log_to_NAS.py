# %%
from pathlib import Path
import glob
import shutil
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

LEGGED_GYM_ROOT_DIR = '/home/terry/taerim/AMP-STMR'
save_log_dir = Path("/media/terry/2884e2c7-1c7d-44b1-ab18-7f2b199dd676/taerim/log")
MOTION_NAME_TO_MOVE = "sidesteps"
logs_names = ["logs50", "logs63", "logs64", "logs65"]
for logs_name in logs_names:
    logs_dir_path = Path(f"{LEGGED_GYM_ROOT_DIR}/{logs_name}")
    if not logs_dir_path.exists(): continue
    for RAND_NORAND in ["STMR"]:
        RAND_NORAND_path = logs_dir_path/RAND_NORAND
        if not RAND_NORAND_path.exists(): continue
        for motion_path in (logs_dir_path/RAND_NORAND).iterdir():
            if motion_path.name == MOTION_NAME_TO_MOVE:
                for robot_path in motion_path.iterdir():
                    for mr_path in robot_path.iterdir():
                        robot = robot_path.name
                        motion = motion_path.name
                        mr = mr_path.name
                        last_path = mr_path/f"{motion}_{robot}_{mr}"
                        if not last_path.exists(): continue
                        for seed_path in last_path.iterdir():
                            old_root_dir = Path(LEGGED_GYM_ROOT_DIR)/logs_name
                            new_root_dir = save_log_dir
                            seed_path_str = seed_path.as_posix()
                            new_seed_path_str = seed_path_str.replace(old_root_dir.as_posix(), new_root_dir.as_posix())
                            new_seed_path = Path(new_seed_path_str)
                            save_path = new_seed_path
                            if save_path.exists():
                                # warning
                                print(f"{save_path} already exists. skipping...")
                            else:
                                print('copying', seed_path, "to", save_path)
                                shutil.copytree(seed_path, save_path)



