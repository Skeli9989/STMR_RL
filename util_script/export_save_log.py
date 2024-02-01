# %%
from pathlib import Path
import glob
import shutil
from legged_gym import LEGGED_GYM_ROOT_DIR


logs_names = ["logs50", "logs63", "logs64"]

for logs_name in logs_names:
    logs_dir_path = Path(f"{LEGGED_GYM_ROOT_DIR}/{logs_name}")
    for motion_path in (logs_dir_path/"STMR").iterdir():
        for robot_path in motion_path.iterdir():
            for mr_path in robot_path.iterdir():
                robot = robot_path.name
                motion = motion_path.name
                mr = mr_path.name
                
                last_path = mr_path/f"{motion}_{robot}_{mr}"
                for seed_path in last_path.iterdir():
                    for date_path in seed_path.iterdir():
                        # move date_path to logs/STMR/{motion}/{robot}/{mr}/{date_path.name}
                        if (date_path/'model_10000.pt').exists():
                            move_path = Path(f"{LEGGED_GYM_ROOT_DIR}/logs_save/STMR/{motion}/{robot}/{mr}/{motion}_{robot}_{mr}/{seed_path.name}/{date_path.name}")
                            # if move_path.exists():
                                # shutil.rmtree(move_path)
                            move_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            if move_path.exists():
                                shutil.rmtree(move_path)
                            # copy date_path to move_path
                            shutil.copytree(date_path, move_path)

# %%
from util_script.checklist import main
main()