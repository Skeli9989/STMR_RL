# %%
from pathlib import Path
import glob
import shutil
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

ROBOT_NAME_TO_DELETE = "albase"

logs_names = ["logs50", "logs63", "logs64", "logs65", 'logs']

for logs_name in logs_names:
    logs_dir_path = Path(f"{LEGGED_GYM_ROOT_DIR}/{logs_name}")
    if not logs_dir_path.exists(): continue

    for RAND_NORAND in ["RAND", "STMR"]:
        RAND_NORAND_path = logs_dir_path/RAND_NORAND
        if not RAND_NORAND_path.exists(): continue
        for motion_path in (logs_dir_path/RAND_NORAND).iterdir():
            for robot_path in motion_path.iterdir():
                if robot_path.name == 'albase':
                    print('deteleting', robot_path)
                    shutil.rmtree(robot_path)

from util_script.checklist import main
main()
