import torch
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator

class CommandProfile:
    def __init__(self, dt, max_time_s=10.):
        self.dt = dt
        self.max_timestep = int(max_time_s / self.dt)
        self.commands = torch.zeros((self.max_timestep, 9))
        self.start_time = 0

    def get_command(self, t):
        timestep = int((t - self.start_time) / self.dt)
        timestep = min(timestep, self.max_timestep - 1)
        return self.commands[timestep, :]

    def get_buttons(self):
        return [0, 0, 0, 0]

    def reset(self, reset_time):
        self.start_time = reset_time

class RCControllerProfile(CommandProfile):
    def __init__(self, dt, state_estimator:StateEstimator):
        super().__init__(dt)
        self.state_estimator = state_estimator
        # self.triggered_commands = {i: None for i in range(4)}  # command profiles for each action button on the controller
        # self.currently_triggered = [0, 0, 0, 0]
        # self.button_states = [0, 0, 0, 0]

    def get_command(self, t, probe=False):

        command = self.state_estimator.get_command()
        RUN = command[0]
        RESET_TIMER = command[1]
        
        return RUN, RESET_TIMER

    # def add_triggered_command(self, button_idx, command_profile):
    #     self.triggered_commands[button_idx] = command_profile

    def get_buttons(self):
        return self.state_estimator.get_buttons()