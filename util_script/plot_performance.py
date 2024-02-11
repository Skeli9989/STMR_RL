# %%
from legged_gym import LEGGED_GYM_ROOT_DIR
from pathlib import Path
from matplotlib import pyplot as plt


def plot_all():
    result_path = Path(LEGGED_GYM_ROOT_DIR) / 'performance' / "STMR"

    for motion_path in result_path.iterdir():
        MOTION = motion_path.name

        for robot_path in motion_path.iterdir():
            ROBOT = robot_path.name
            if 'base' in ROBOT:
                raw_ROBOT = ROBOT.split('base')[0]
            
            for mr_path in robot_path.iterdir():
                MR = mr_path.name

                for experiment_path in mr_path.iterdir():
                    # if experiment_path is not dir
                    if not experiment_path.is_dir():
                        continue

                    experiment_name = experiment_path.name

                    for seed_path in experiment_path.iterdir():
                        seed = seed_path.name

                        # read json file
                        json_path = seed_path / 'pose_all.json'


                        if not json_path.exists():
                            print(f"{ROBOT}_{MR}_{seed} does not exist")
                            continue
                        import json
                        with open(json_path) as f:
                            data = json.load(f)
                        
                        dtw_distance = data['dtw_distance']
                        # len(xrange) is len(dtw_distance) and interval is 50
                        xrange = [i*50 for i in range(len(dtw_distance))]
                        plt.plot(xrange, dtw_distance, label=f"{ROBOT}_{MR}_{MOTION}_{seed}")
            plt.legend()
            save_name = Path(f"{LEGGED_GYM_ROOT_DIR}/performance/fig/{ROBOT}/{MOTION}/{ROBOT}_{MOTION}.png")
            save_name.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_name)
            plt.show()


def plot_last():
    result_path = Path(LEGGED_GYM_ROOT_DIR) / 'performance' / "STMR"

    res_dict = {}
    for motion_path in result_path.iterdir():
        MOTION = motion_path.name


        for robot_path in motion_path.iterdir():
            ROBOT = robot_path.name
            if 'base' in ROBOT:
                raw_ROBOT = ROBOT.split('base')[0]
            else: 
                raw_ROBOT = ROBOT
            
            row_column_name = f"{raw_ROBOT}_{MOTION}"
            if MOTION not in res_dict.keys():
                res_dict[row_column_name] = {}
            
            
            for mr_path in robot_path.iterdir():
                MR = mr_path.name

                for experiment_path in mr_path.iterdir():
                    # if experiment_path is not dir
                    if not experiment_path.is_dir():
                        continue

                    experiment_name = experiment_path.name

                    for seed_path in experiment_path.iterdir():
                        seed = seed_path.name

                        # read json file
                        json_path = seed_path / 'pose_1k.json'


                        if not json_path.exists():
                            print(f"{ROBOT}_{MR}_{seed} does not exist")
                            continue
                        import json
                        with open(json_path) as f:
                            data = json.load(f)
                        
                        dtw_distance = data['dtw_distance']
                        # len(xrange) is len(dtw_distance) and interval is 50
                        

                        if MR not in res_dict[row_column_name].keys():
                            res_dict[row_column_name][MR] = []
                        res_dict[row_column_name][MR].append(dtw_distance[-1])
    
    # plot table using pandas
    import pandas as pd
    from pandas.plotting import table
    import numpy as np
    df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['dtw_distance'])

    res_dict_mean = {}
    res_dict_std  = {}
    for row,col in res_dict.items():
        if row not in res_dict_mean.keys():
            res_dict_mean[row] = {}
            res_dict_std[row] = {}
        for MR, distance_ls in col.items():
            res_dict_mean[row][MR] = np.mean(distance_ls)
            res_dict_std[row][MR] = np.std(distance_ls)
    
    res_dict_dict = dict(
        mean = res_dict_mean,
        std = res_dict_std
    )
    for key, res_dict in res_dict_dict.items():
        
        df = pd.DataFrame(res_dict.values(), index=res_dict.keys())
        for col in df.columns:
            df[col] = df[col].round(3)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        tab = table(ax, df, loc='center', cellLoc='center')

        # Style the table
        tab.auto_set_font_size(False)
        tab.set_fontsize(12)
        tab.scale(1.2, 1.2)

        # Save the image
        plt.tight_layout()
        plt.title("DTW distance", fontsize=16, pad=20)

        save_name = Path(f"{LEGGED_GYM_ROOT_DIR}/performance/table/dtw_distance_{key}.png")
        save_name.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_name)

plot_last()