# %%
# Extract contact boolean from .tar
import numpy as np
from pathlib import Path
from legged_gym import LEGGED_GYM_ROOT_DIR

MOTIONS = ['go1trot', 'hopturn', 'pace0', 'pace1', 'trot0', 'trot1']

MOTION = 'go1trot'
for MOTION in MOTIONS:
    path = LEGGED_GYM_ROOT_DIR/Path(f"datasets/{MOTION}/{MOTION}_tar.npz")

    data = np.load(path, allow_pickle=True)


    import json

    contact_dict = dict(
        FR = data['contact_schedule_src'][0,:].tolist(),
        FL = data['contact_schedule_src'][1,:].tolist(),
        RR = data['contact_schedule_src'][2,:].tolist(),
        RL = data['contact_schedule_src'][3,:].tolist(),
    )

    # save to json
    savename = LEGGED_GYM_ROOT_DIR/Path(f"datasets/{MOTION}/{MOTION}_contact.json")
    with open(savename, 'w') as fp:
        json.dump(contact_dict, fp)