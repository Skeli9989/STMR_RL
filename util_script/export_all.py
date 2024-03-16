# %%
from util_script.export_onnx import export_onnx
from util_script.export_json import export_json_files

ROBOT = "go1base".lower()
if 'base' in ROBOT:
	raw_robot_name = ROBOT.split("base")[0]
else:
	raw_robot_name = ROBOT
NO_RAND = False
MOTION = "trot0"
SEED = 5
MR = "STMR"
task = f"{ROBOT}_{MR}_{MOTION}"	

export_onnx(task, seed=SEED, NO_RAND=NO_RAND, device = 'cpu')
export_json_files(raw_robot_name, MOTION, MR, PLOT=True)
print('done')
# %%
	