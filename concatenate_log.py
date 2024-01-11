# %%
from pathlib import Path
import glob
import shutil

zip_path = Path("logs.zip")
unzip_path = Path("logs_tmp")

import zipfile
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_path)

# %%
# file structure of unzip_path
# unzip_path/STMR/{motion}/{robot}/{MR}/{motion}_{robot}_{MR}/*
# mv unzip_path/logs/STMR/{motion}/{robot}/{MR}/{motion}_{robot}_{MR}/* logs/STMR/{motion}/{robot}/{MR}/{motion}_{robot}_{MR}/    
for motion_path in (unzip_path/"logs"/"STMR").iterdir():
    for robot_path in motion_path.iterdir():
        for mr_path in robot_path.iterdir():
            robot = robot_path.name
            motion = motion_path.name
            mr = mr_path.name

            last_path = mr_path/f"{motion}_{robot}_{mr}"

            for date_path in last_path.iterdir():
                # move date_path to logs/STMR/{motion}/{robot}/{mr}/{date_path.name}
                if (date_path/'model_10000.pt').exists():
                    move_path = Path(f"logs/STMR/{motion}/{robot}/{mr}/{motion}_{robot}_{mr}/{date_path.name}")
                    
                    if move_path.exists():
                        shutil.rmtree(move_path)
                    move_path.parent.mkdir(parents=True, exist_ok=True)                    
                    date_path.rename(move_path)

# %%
# remove unzip_path
shutil.rmtree(unzip_path)

