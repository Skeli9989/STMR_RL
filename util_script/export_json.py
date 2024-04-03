
AMP_DATASET_DIR = "AMP_dataset"

from mjpc import ROOT_DIR as ROOT_DIR
import mujoco
from mujoco_viewer.mujoco_viewer import MujocoViewer
import numpy as np
from numpy.linalg import norm

from toolbox.resources import get_asset_dict, ASSET_SRC
from mjmr.util import reset, get_mr_info, get_xml_path, plot_contact_schedule, get_vel_contact_boolean, get_mujoco_contact_boolean,get_vel_contact_boolean_from_pos, get_key_id, output_amp_motion
from mjmr.task.Quadruped.info import QuadrupedRetargetInfo as RetargetInfo
from mjmr.motion_holder import IKTargetHolder
from mjmr.agent import MJMR


from mjpc.ILQG.ilqg import ILQG
from mjpc.motion_holder import MPCMotionHolder
from mjpc.recorder import Recorder
from mjpc.time_scaler import TimeScaler
from mjpc.util import reset, scale_interval, get_xml_path, scale_multi_interval, make_mpc_tar_dict
from mjpc.task.Quadruped.info import QuadrupedMPCInfo as MPCInfo
from mjpc.task.Quadruped.sensor_callback import quadruped_residual as compute_residual
from mjpc.task.Quadruped.play import quadruped_play as play
from mjpc.util import get_motion_specific_config

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
import json

def export_json_files(ROBOT, MOTION, MR, PLOT=False):


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

	# %%
	motion_config = get_motion_specific_config(MOTION, ROBOT)
	FLIP_REAR_LEG = motion_config['FLIP_REAR_LEG']
	preference = motion_config['preference']

	mr_info   = RetargetInfo(model, data)
	mpc_info = MPCInfo(model, data, preference=preference)

	pybullet_filename = f"{ROOT_DIR}/../{AMP_DATASET_DIR}/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_raw.txt"
	from toolbox.read import read_json
	motion_dict = read_json(pybullet_filename)

	qpos_np = np.array(motion_dict['Frames'])[:,:19]
	dt = motion_dict['FrameDuration']
	qpos_np[:,[3,4,5,6]] = qpos_np[:,[6,3,4,5]]
	qpos_np[:,7:] += model.qpos0[7:]

	for idx,q in enumerate(qpos_np):
		if idx % 10 == 0:
			data.qpos[:] = q
			mujoco.mj_forward(model, data)
			viewer.render()

	if ROBOT == 'al':
		mr_info.size.contact_criteria = mr_info.size.foot * 1.5
	elif ROBOT == 'go1':
		mr_info.size.contact_criteria = mr_info.size.foot * 0.95

	thres_dist = mr_info.size.contact_criteria
	mj_contact_boolean = get_mujoco_contact_boolean(model, data, mr_info, qpos_np, thres_dist)

	if PLOT:
		plot_contact_schedule(*mj_contact_boolean)
	# %%
	frame_number = len(qpos_np)

	time_array = np.arange(0, frame_number, 1) * dt

	contact_json = {'data':[]}
	for time_, contact in zip(time_array, mj_contact_boolean.T):
		contact = contact.tolist()
		contact_per_foot = {
			"FR": contact[0],
			"FL": contact[1],
			"RR": contact[2],
			"RL": contact[3],
		}
		contact_json['data'].append([time_, contact_per_foot])

	save_name = f"{ROOT_DIR}/../{AMP_DATASET_DIR}/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_contact.json"
	with open(save_name, 'w') as f:
		json.dump(contact_json, f)

	# %%
	isaac_filename = f"{ROOT_DIR}/../{AMP_DATASET_DIR}/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_processed/{MOTION}_{ROBOT}_{MR}.txt"
	motion_dict = read_json(isaac_filename)
	joint_values = np.array(motion_dict['Frames'])[:,7:19]

	motion_frame_json = {'data':[]}
	for time_, joint in zip(time_array, joint_values):
		motion_frame_json['data'].append([time_, *joint])

	save_name = f"{ROOT_DIR}/../{AMP_DATASET_DIR}/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_joint_values.json"
	with open(save_name, 'w') as f:
		json.dump(motion_frame_json, f)
