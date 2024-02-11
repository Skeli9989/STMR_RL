# %%
# STMR checklist
from legged_gym import LEGGED_GYM_ROOT_DIR

def main():
    from pathlib import Path

    dataset_path = Path(f'{LEGGED_GYM_ROOT_DIR}/datasets')

    robots = ['go1', 'a1', 'al']
    motions = ['go1trot', 'hopturn', 'pace0', 'pace1', 'sidesteps', 'trot0', 'trot1', 'videowalk0', 'videowalk1']

    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas.plotting import table

    MRs = ['NMR', 'SMR', 'TMR', "STMR", "TO", "AMP"]
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
        Path(f'{LEGGED_GYM_ROOT_DIR}/checklist/motion_checklist').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{LEGGED_GYM_ROOT_DIR}/checklist/motion_checklist/{MR}.png', bbox_inches='tight', pad_inches=0.05)
        # plt.show()

    # RL checklist
    import glob

    log_path = Path(f'{LEGGED_GYM_ROOT_DIR}/logs')
    data = {}

    break_flag = False
    for MR in MRs:
        data[MR] = {}
        for motion in motions:
            data[MR][motion] = {}
            for robot in robots:
                robot = robot+"base"
                paths=log_path/"STMR"/motion/robot/MR/f"{motion}_{robot}_{MR}"

                if robot not in data[MR][motion].keys():
                    data[MR][motion][robot] = ""

                if not paths.exists():
                    continue
                for seed_path in paths.iterdir():
                    seed_name = seed_path.name

                    if glob.glob(str(paths/"**/model_10000.pt"), recursive=True):
                        seed_num = seed_name.split("seed")[-1]
                        data[MR][motion][robot] += f"{seed_num}/"
                
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
        Path(f'{LEGGED_GYM_ROOT_DIR}/checklist/RL_checklist').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{LEGGED_GYM_ROOT_DIR}/checklist/RL_checklist/{MR}.png', bbox_inches='tight', pad_inches=0.05)

if __name__ == '__main__':
    main()