from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.envs.a1_robot import env_builder
from legged_gym.envs.a1_robot import locomotion_gym_config
from legged_gym.envs.a1_robot import robot_config
from legged_gym.envs.a1_robot import a1

import numpy as np
import torch


_convert_obs_dict_to_tensor = lambda obs, device: torch.tensor(np.concatenate([
        obs["ProjectedGravity"], obs["FakeCommand"], obs["MotorAngle"],
        obs["MotorVelocity"], obs["LastAction"]]), device=device).float()


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.get_commands_from_joystick = False
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False

    train_cfg.runner.amp_num_preload_transitions = 1

    # prepare environment
    args.headless = True
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = True
    sim_params.allow_knee_contact = True
    sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION
    sim_params.num_action_repeat = int(1. / (env_cfg.sim.dt * env_cfg.control.decimation))
    sim_params.enable_action_filter = False
    sim_params.torque_limits = 40.0
    sim_params.enable_clip_motor_commands = False
    sim_params.enable_action_interpolation = False

    pyb_env = env_builder.build_env_isaac(
        sim_params=sim_params,
        default_pose=a1.INIT_MOTOR_ANGLES,
        obs_scales=env_cfg.normalization.obs_scales,
        action_scale=env_cfg.control.action_scale,
        use_real_robot=False)
    # Dict -> Tensor.
    obs = _convert_obs_dict_to_tensor(pyb_env.reset(), ppo_runner.device)


    for i in range(int(10e4)):
        # Evaluate policy and act.
        actions = policy(obs.detach())
        obs, _, _, _ = pyb_env.step(actions.detach().cpu().numpy())

        # Update commands to get latest from joystick.
        env.compute_observations()
        commands = np.squeeze(env.obs_buf[..., 3:6].detach().cpu().numpy())
        obs["FakeCommand"] = commands

        # Dict -> Tensor.
        obs = _convert_obs_dict_to_tensor(obs, ppo_runner.device)

if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    play(args)
