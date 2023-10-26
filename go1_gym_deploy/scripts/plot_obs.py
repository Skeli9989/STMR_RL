# %%
import numpy as np
obs_array = np.loadtxt("/home/taerim/obs_list.txt")

# %%

ROBOT = 'go1'.lower()
MR = "TMR"
MOTION = "hopturn"

import mujoco
from mujoco_viewer import MujocoViewer
from toolbox.resources import get_asset_dict, ASSET_SRC
xml_dict = get_asset_dict(ASSET_SRC.CUSTOM, XML=True)
if ROBOT.lower() == "go1":
    xml_path = xml_dict['GO1_Motion_Task']


model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
data  = mujoco.MjData(model)

try:
    viewer.close()
except Exception:
    pass
viewer = MujocoViewer(
   model,data,mode='window',title="PlotObs",
   width=1200,height=800,hide_menus=True
   )

# %%
from mjpc.util import plot_arrow
from go1_gym_deploy.utils.motion_holder import MotionHolder
from go1_gym_deploy import BASEDIR
from go1_gym_deploy.utils.cfg import Cfg

cfg = Cfg()
MOTION_FILE = BASEDIR / f"run/{MOTION}/{MR}/{MOTION}_{ROBOT}_{MR}.txt"
motion_file = str(MOTION_FILE)


joint_names = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", ]

default_dof_pos = np.array([
            cfg.init_state.default_joint_angles[name] for name in joint_names])


for timestep, obs_ in enumerate(obs_array[:40]):
    time_ = timestep * 0.03
    data.qpos[7:] = obs_[3:3+12]/cfg.normalization.obs_scales.dof_pos + default_dof_pos
    mujoco.mj_forward(model,data)
    for _ in range(20):
        plot_arrow(viewer, p=data.qpos[:3], uv=obs_[:3])
        viewer.render()


# %%
motion_holder = MotionHolder(motion_file)
for timestep, obs_ in enumerate(obs_array[:40]):
    time_ = timestep * 0.03
    data.qpos[7:] = motion_holder.get_q(time_)
    mujoco.mj_forward(model,data)
    for _ in range(20):
        # plot_arrow(viewer, p=data.qpos[:3], uv=obs_[:3])
        viewer.render()

# %%
print(obs_array[:,-1])


# obs = np.concatenate([
#     projected_gravity,
#     (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
#     dof_vel * self.obs_scales.dof_vel,
#     self.actions,
#     np.array([deploy_time]),
# ])