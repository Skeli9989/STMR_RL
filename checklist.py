# %%
# STMR checklist

from pathlib import Path

dataset_path = Path('datasets')

robots = ['go1', 'a1', 'al']
motions = ['go1trot', 'hopturn', 'pace0', 'pace1', 'sidesteps', 'trot0', 'trot1', 'videowalk0', 'videowalk1']

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

MRs = ['NMR', 'SMR', 'TMR', "STMR", "TO"]
data = {}

for MR in MRs:
    data[MR] = {}
    for motion in motions:
        data[MR][motion] = {}
        for robot in robots:
            path = dataset_path / motion / robot/ MR / f'{motion}_{robot}_{MR}_raw.txt'
            if path.exists():
                data[MR][motion][robot] = 'O'
            else:
                data[MR][motion][robot] = 'X'

    df = pd.DataFrame(data[MR])
    df.index = robots
    fig, ax = plt.subplots(figsize=(8, 4)) 
    ax.axis('off')
    tab = table(ax, df, loc='center', cellLoc='center')
    
    # Style the table
    tab.auto_set_font_size(False)
    tab.set_fontsize(12)
    tab.scale(1.2, 1.2)

    # Save the image
    plt.title(f'{MR}_Motion_checklist', fontsize=16, pad=20)
    plt.savefig(f'checklist/{MR}_Motion_checklist.png', bbox_inches='tight', pad_inches=0.05)
    plt.show()

# %%
# RL checklist
import glob

log_path = Path('logs')
data = {}

break_flag = False
for MR in MRs:
    data[MR] = {}
    for motion in motions:
        data[MR][motion] = {}
        for robot in robots:
            paths=log_path/"STMR"/motion/robot/MR/f"{motion}_{robot}_{MR}"

            # if "model_10000.pt" exists in recusive paths
            if glob.glob(str(paths/"**/model_8000.pt"), recursive=True):
                data[MR][motion][robot] = 'O'
            else:
                data[MR][motion][robot] = 'X'

            
    df = pd.DataFrame(data[MR])
    df.index = robots
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    tab = table(ax, df, loc='center', cellLoc='center')

    # Style the table
    tab.auto_set_font_size(False)
    tab.set_fontsize(12)
    tab.scale(1.2, 1.2)

    # Save the image
    plt.title(f'{MR}_RL_checklist', fontsize=16, pad=20)
    plt.savefig(f'checklist/{MR}_RL_checklist.png', bbox_inches='tight', pad_inches=0.05)
# %%
