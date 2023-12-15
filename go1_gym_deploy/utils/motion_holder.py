import numpy as np
import json

def load_motion_file(file_path):
    if type(file_path) is list:
        print("Warning: file_path is a list, using the first element")
        file_path = file_path[0]
    with open(file_path) as f:
        return json.load(f)

class MotionHolder():
    def __init__(self, motion_file):
        self.motion_src = load_motion_file(motion_file)
        self.q_src = np.array(self.motion_src["Frames"])[:,3+4:3+4+12]
        self.dt = self.motion_src["FrameDuration"]
        self.time_array = np.arange(len(self.q_src)) * self.dt
    
        self.max_time = self.time_array[-1]
        
    def get_q(self, time_):
        idx = np.searchsorted(self.time_array, time_)
        
        if idx == 0:
            # print(0)
            return self.q_src[0]
        elif idx == len(self.time_array):
            raise ValueError("time_ is out of range")
        else:
            idx_fr = idx-1
            t1 = self.time_array[idx_fr]
            t2 = self.time_array[idx]
            q1 = self.q_src[idx_fr]
            q2 = self.q_src[idx]
            alpha = (time_ - t1) / (t2 - t1)
            return (1 - alpha) * q1 + alpha * q2
    
    def get_qvel(self, time_):
        q = self.get_q(time_)
        q_af = self.get_q(time_ + self.dt)
        return (q_af - q) / self.dt
    
    def get_batch_q(self, times_):
        idx = np.searchsorted(self.time_array, times_)
        
        bsize = len(times_)
        q_dim = len(self.q_src[0])
        
        res = np.zeros((bsize, q_dim))
        
        
        first_mask = idx==0
        res[first_mask] = self.q_src[0]
        
        last_mask = idx==len(self.time_array)
        res[last_mask] = self.q_src[-1]
        
        neither_mask = np.logical_and(~ first_mask, ~ last_mask)
        idx_neither = idx[neither_mask]
        
        t1 = self.time_array[idx_neither-1]
        t2 = self.time_array[idx_neither]
        q1 = self.q_src[idx_neither-1]
        q2 = self.q_src[idx_neither]
        time_neither = times_[neither_mask]
        alpha = ((time_neither - t1) / (t2 - t1)).reshape(-1,1)
        res[neither_mask] = (1 - alpha) * q1 + alpha * q2
        
        return res

    def get_batch_qvel(self, times_):
        q = self.get_batch_q(times_)
        q_af = self.get_batch_q(times_ + self.dt)
        return (q_af - q) / self.dt