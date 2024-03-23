# %%
from legged_gym import LEGGED_GYM_ROOT_DIR
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

def plot_all():
    result_path = Path(LEGGED_GYM_ROOT_DIR) / 'performance' / "STMR"

    for motion_path in result_path.iterdir():
        MOTION = motion_path.name

        for robot_path in motion_path.iterdir():
            ROBOT = robot_path.name
            if 'base' in ROBOT:
                raw_ROBOT = ROBOT.split('base')[0]
            
            plt.figure(figsize=(10,10))
            plt.title(f"{raw_ROBOT}_{MOTION}")
            plt.tight_layout()
            for mr_path in robot_path.iterdir():
                MR = mr_path.name

                distance_per_seed_ls= []
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
                        try:
                            with open(json_path) as f:
                                data = json.load(f)
                        except:
                            print(f"failed to read {json_path}")
                            continue
                        dtw_distance = data['dtw_distance']
                        
                        if len(distance_per_seed_ls) >0:
                            if len(dtw_distance) != len(distance_per_seed_ls[-1]):
                                print(f"somthing wrong with {ROBOT}_{MR}_{MOTION}_{seed}")
                                continue
                                # if MR == "AMP":
                                    # continue
                                # else:
                                    # raise Exception(f"somthing wrong with {ROBOT}_{MR}_{MOTION}_{seed}")
                        
                        distance_per_seed_ls.append(dtw_distance)
                        # len(xrange) is len(dtw_distance) and interval is 50

                if distance_per_seed_ls == []:
                    continue

                distance_per_seed = np.array(distance_per_seed_ls)
                mean = np.mean(distance_per_seed, axis=0)
                std  = np.std(distance_per_seed, axis=0)
                xrange = [i*50 for i in range(len(mean))]
                plt.plot(xrange, mean, label=f"{MR}")
                plt.fill_between(xrange, mean-std, mean+std, alpha=0.3)
                plt.legend()
            save_name = Path(f"{LEGGED_GYM_ROOT_DIR}/performance/fig/{ROBOT}/{MOTION}.png")
            save_name.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_name)
            print(f"save {save_name}")
                # plt.show()


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


    # res_dict_mean = {}
    # res_dict_std  = {}
    res_dict_save = {}
    for row,col in res_dict.items():
        if row not in res_dict_save.keys():
            res_dict_save[row] = {}
        for MR, distance_ls in col.items():
            mean = np.round(1000*np.mean(distance_ls),1)
            std  = np.round(1000*np.std(distance_ls),1)
            res_dict_save[row][MR] = f"{mean}±{std}"
    
    df = pd.DataFrame(res_dict_save.values(), index=res_dict_save.keys())
    desired_col_order = ["NMR", "AMP", "TO", "STMR"]
    df = df[desired_col_order]

    desired_row_order = []
    for motion in ['trot0', 'trot1', 'pace0', 'pace1', 'sidesteps', 'hopturn']:
        for robot_name in ['go1', 'a1', 'al']:
            desired_row_order.append(f"{robot_name}_{motion}")

    desired_row_order = [label for label in desired_row_order if label in df.index]
    df = df.reindex(desired_row_order)

    print(df)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    tab = table(ax, df, loc='center', cellLoc='center')

    # Style the table
    tab.auto_set_font_size(False)
    tab.set_fontsize(12)
    tab.scale(1.2, 1.2)

    # Save the image
    plt.tight_layout()
    plt.title("DTW distance", fontsize=16, pad=20)

    save_name = Path(f"{LEGGED_GYM_ROOT_DIR}/performance/table/dtw_distance.png")
    save_name.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_name)

plot_last()