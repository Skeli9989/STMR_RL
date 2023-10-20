import math
import select
import threading
import time

import numpy as np
import lcm

from go1_gym_deploy.lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from go1_gym_deploy.lcm_types.state_estimator_lcmt import state_estimator_lcmt
from go1_gym_deploy.lcm_types.rc_command_lcmt import rc_command_lcmt

def get_rpy_from_quaternion(q):
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    p = np.arcsin(2 * (w * y - z * x))
    y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([r, p, y])


def get_rotation_matrix_from_rpy(rpy):
    """
    Get rotation matrix from the given quaternion.
    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]
    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    r, p, y = rpy
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]
                    ])

    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]
                    ])

    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]
                    ])

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot


class GO1StateEstimator():
    def __init__(self, lc:lcm.LCM):
        self.lc = lc
        
        # Constants
        self.joint_idxs = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.contact_idxs = [1, 0, 3, 2]
        
        # Make buffer
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.tau_est = np.zeros(12)
        
        self.euler = np.zeros(3)
        self.R = np.eye(3)
        self.buf_idx = 0

        self.smoothing_length = 12
        self.deuler_history = np.zeros((self.smoothing_length, 3))
        self.dt_history = np.zeros((self.smoothing_length, 1))
        self.euler_prev = np.zeros(3)
        self.timuprev = time.time()
        self.smoothing_ratio = 0.2

        self.RUN = 0 # 0: stop, 1: run

        # Key status in joystick
        self.mode = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]

        self.left_upper_switch = 0                  # ?
        self.left_upper_switch_pressed = 0
        
        self.left_lower_left_switch = 0             # L1
        self.left_lower_left_switch_pressed = 0
        
        self.left_lower_right_switch = 0            # L2
        self.left_lower_right_switch_pressed = 0
        
        self.right_upper_switch = 0                 # ?
        self.right_upper_switch_pressed = 0
        
        self.right_lower_left_switch = 0            # R1
        self.right_lower_left_switch_pressed = 0
        
        self.right_lower_right_switch = 0           # R2
        self.right_lower_right_switch_pressed = 0

        self.contact_state = np.ones(4)
        
        self.imu_subscription = self.lc.subscribe("state_estimator_data", self._imu_cb)
        self.legdata_state_subscription = self.lc.subscribe("leg_control_data", self._legdata_cb)
        self.rc_command_subscription = self.lc.subscribe("rc_command", self._rc_command_cb)

    def _legdata_cb(self, channel, data):
        msg = leg_control_data_lcmt.decode(data)
        self.joint_pos = np.array(msg.q)
        self.joint_vel = np.array(msg.qd)
        self.tau_est = np.array(msg.tau_est)

    def _imu_cb(self, channel, data):
        msg = state_estimator_lcmt.decode(data)

        self.euler = np.array(msg.rpy)

        self.R = get_rotation_matrix_from_rpy(self.euler)

        self.contact_state = 1.0 * (np.array(msg.contact_estimate) > 200)

        self.deuler_history[self.buf_idx % self.smoothing_length, :] = msg.rpy - self.euler_prev
        self.dt_history[self.buf_idx % self.smoothing_length] = time.time() - self.timuprev

        self.timuprev = time.time()

        self.buf_idx += 1
        self.euler_prev = np.array(msg.rpy)
        
    def _rc_command_cb(self, channel, data):
        msg = rc_command_lcmt.decode(data)

        self.left_upper_switch_pressed = ((msg.left_upper_switch and not self.left_upper_switch) or self.left_upper_switch_pressed)
        self.left_lower_left_switch_pressed = ((msg.left_lower_left_switch and not self.left_lower_left_switch) or self.left_lower_left_switch_pressed)
        self.left_lower_right_switch_pressed = ((msg.left_lower_right_switch and not self.left_lower_right_switch) or self.left_lower_right_switch_pressed)
        self.right_upper_switch_pressed = ((msg.right_upper_switch and not self.right_upper_switch) or self.right_upper_switch_pressed)
        self.right_lower_left_switch_pressed = ((msg.right_lower_left_switch and not self.right_lower_left_switch) or self.right_lower_left_switch_pressed)
        self.right_lower_right_switch_pressed = ((msg.right_lower_right_switch and not self.right_lower_right_switch) or self.right_lower_right_switch_pressed)

        self.mode = msg.mode
        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.left_upper_switch = msg.left_upper_switch
        self.left_lower_left_switch = msg.left_lower_left_switch
        self.left_lower_right_switch = msg.left_lower_right_switch
        self.right_upper_switch = msg.right_upper_switch
        self.right_lower_left_switch = msg.right_lower_left_switch
        self.right_lower_right_switch = msg.right_lower_right_switch

    def get_command(self):
        if self.left_lower_right_switch_pressed:    # L2
            self.RUN = (self.RUN + 1) % 2
        
        if self.right_lower_right_switch_pressed:   # R2
            RESET_TIMER = 1
        
        return self.RUN, RESET_TIMER


    def get_body_angular_vel(self):
        self.body_ang_vel = self.smoothing_ratio * np.mean(self.deuler_history / self.dt_history, axis=0) + (
                    1 - self.smoothing_ratio) * self.body_ang_vel
        return self.body_ang_vel

    def get_gravity_vector(self):
        grav = np.dot(self.R.T, np.array([0, 0, -1]))
        return grav

    def get_contact_state(self):
        return self.contact_state[self.contact_idxs]

    def get_rpy(self):
        return self.euler

    def get_buttons(self):
        return np.array([self.left_lower_left_switch, self.left_upper_switch, self.right_lower_right_switch, self.right_upper_switch])

    def get_dof_pos(self):
        return self.joint_pos[self.joint_idxs]

    def get_dof_vel(self):
        return self.joint_vel[self.joint_idxs]

    def get_tau_est(self):
        return self.tau_est[self.joint_idxs]

    def get_yaw(self):
        return self.euler[2]
    
    def poll(self, cb=None):
        t = time.time()
        try:
            while True:
                timeout = 0.01
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                if rfds:
                    # print("message received!")
                    self.lc.handle()
                    # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
                else:
                    continue
                    # print(f'waiting for message... Freq {1. / (time.time() - t)} Hz'); t = time.time()
                #    if cb is not None:
                #        cb()
        except KeyboardInterrupt:
            pass

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()

    def close(self):
        self.lc.unsubscribe(self.legdata_state_subscription)