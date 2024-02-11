# %%
ROBOT = "a1".lower()
MOTION = 'hopturn'
seed = 0
MR = "STMR"
# MR = "NMR"
# MR = "STMR"
# MR = "AMP"

GET_ALL = False

# %%
import mujoco
from mujoco_viewer.mujoco_viewer import MujocoViewer
import numpy as np
from numpy.linalg import norm

from toolbox.resources import get_asset_dict, ASSET_SRC
from mjmr.util import reset, get_mr_info, get_xml_path, plot_contact_schedule, get_vel_contact_boolean, get_mujoco_contact_boolean,get_vel_contact_boolean_from_pos, get_key_id, output_amp_motion
from mjmr.task.Quadruped.info import QuadrupedRetargetInfo as RetargetInfo

from MotionBO.MotionBO import compute_cost
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from matplotlib import pyplot as plt
import copy
import time
from pathlib import Path
import shutil

xml_path = get_xml_path(ROBOT)
model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
data  = mujoco.MjData(model)
reset(model, data)

try:
   viewer.close()
except Exception:
   pass

viewer = MujocoViewer(
   model,data,mode='window',title="MPC",
   width=1200,height=800,hide_menus=True
   )
mr_info   = RetargetInfo(model, data)

# %%
from mjpc.task.Quadruped.play import plot_skeleton
from mjpc.task.Quadruped.info import QuadrupedMPCInfo as MPCInfo

mpc_info = MPCInfo(model, data)

# %%
from legged_gym import LEGGED_GYM_ROOT_DIR
from fastdtw import fastdtw
import numpy as np
from scipy.spatial.distance import cityblock
# load motion
ROBOT_base = ROBOT + 'base'

if GET_ALL:
    path = Path(LEGGED_GYM_ROOT_DIR)/f"performance/STMR/{MOTION}/{ROBOT_base}/{MR}/{MOTION}_{ROBOT_base}_{MR}/seed{seed}/pose_all.json"
else:
    path = Path(LEGGED_GYM_ROOT_DIR)/f"performance/STMR/{MOTION}/{ROBOT_base}/{MR}/{MOTION}_{ROBOT_base}_{MR}/seed{seed}/pose_1k.json"
import json
# load json
with open(path) as json_file:
    motion_data = json.load(json_file)

target_qpos_array = np.array(motion_data['target'])
deploy_qpos_array = np.array(motion_data['deploy'])

target_qpos_array.shape, deploy_qpos_array.shape

reset(model,data)

site_ids = [
    mr_info.id.trunk_site,
    mr_info.id.FR_hip_site, mr_info.id.FR_calf_site, mr_info.id.FR_foot_site,
    mr_info.id.FL_hip_site, mr_info.id.FL_calf_site, mr_info.id.FL_foot_site,
    mr_info.id.RR_hip_site, mr_info.id.RR_calf_site, mr_info.id.RR_foot_site,
    mr_info.id.RL_hip_site, mr_info.id.RL_calf_site, mr_info.id.RL_foot_site,
    ]

# site_ids = [
#     mr_info.id.trunk_site,
#     mr_info.id.FR_hip_site, mr_info.id.FR_calf_site,
#     mr_info.id.FL_hip_site, mr_info.id.FL_calf_site,
#     mr_info.id.RR_hip_site, mr_info.id.RR_calf_site,
#     mr_info.id.RL_hip_site, mr_info.id.RL_calf_site,
#     ]

mujoco_idx = [3,4,5,6, 7,8,9, 10,11,12, 13,14,15, 16,17,18]
amp_idx    = [6,3,4,5, 10,11,12, 7,8,9, 16,17,18, 13,14,15]

target_qpos_total = np.array(motion_data['target'])
deploy_qpos_total = np.array(motion_data['deploy'])

target_qpos_total[:,:,mujoco_idx] = target_qpos_total[:,:,amp_idx].copy()
deploy_qpos_total[:,:,mujoco_idx] = deploy_qpos_total[:,:,amp_idx].copy()

target_qpos_total[:,:, 7:] +=  model.qpos0[7:] 
deploy_qpos_total[:,:, 7:] +=  model.qpos0[7:]

model_number = len(target_qpos_total)
frame_number = len(target_qpos_total[0])

print(model_number)

if GET_ALL:
    render_every = 10
else:
    render_every = 1
key_point_error_ls = []
for model_i in range(model_number):
    # model_i = -1
    target_qpos_array = target_qpos_total[model_i]
    deploy_qpos_array = deploy_qpos_total[model_i]

    target_site_ls = []
    deploy_site_ls = []
    for frame_i in range(frame_number):

        for _ in range(1):
            target_qpos = target_qpos_array[frame_i]
            deploy_qpos = deploy_qpos_array[frame_i]

            data.qpos[:] = target_qpos
            mujoco.mj_forward(model, data)
            for site_i in site_ids:
                target_site_ls.append(data.site_xpos[site_i].copy())
            # viewer.render()

            if frame_i%render_every == 0:
                plot_skeleton(model, data, mpc_info, viewer, rgba = [1,0,0,0.5])

            data.qpos[:] = deploy_qpos
            mujoco.mj_forward(model, data)
            for site_i in site_ids:
                deploy_site_ls.append(data.site_xpos[site_i].copy())
            if frame_i%render_every == 0:
                viewer.render()    

    target_site_array = np.array(target_site_ls).reshape(frame_number, -1)
    deploy_site_array = np.array(deploy_site_ls).reshape(frame_number, -1)

    distance, path = fastdtw(target_site_array, deploy_site_array, dist=cityblock)
    key_point_error = distance/frame_number/len(site_ids)

    # key_point_error = np.mean(np.abs(np.array(target_site_ls) - np.array(deploy_site_ls)), axis=1)
    # key_point_error = np.max(key_point_error, axis=0)
    
    key_point_error_ls.append(key_point_error)
    print(key_point_error)
    # break
    
# L1 loss between target and deploy site
from matplotlib import pyplot as plt
plt.plot(key_point_error_ls)

# %%
