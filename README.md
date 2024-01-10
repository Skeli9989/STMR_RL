Spatial-Temporal Motion Retargeting

1. Install pytorch 1.13 with cuda-11.6:
`pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`
2. Install Isaac Gym
   - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
3. Install rsl_rl (PPO implementation)
   - Clone this repository
   -  `cd AMP_for_hardware/rsl_rl && pip install -e .`
4. Install legged_gym
   - `cd ../ && pip install -e .`

### Usage in simulation ###
1. Train:
    `python legged_gym/scripts/train.py --task={ROBOT}_{MR}_{MOTION}`
    -  ROBOT: go1, a1, al (i.e. aliengo)
    -  MR: NMR, SMR, TMR, STMR, TO
    -  MOTION: trot0, trot1, pace0, pace1, hopturn, sidesteps, videowalk0, videowalk1
2. Play target motion:
`python legged_gym/scripts/play_target_motion.py --task={ROBOT}_{MR}_{MOTION}`
3. Play policy
`python legged_gym/scripts/play.py --task={ROBOT}_{MR}_{MOTION}`
    - This also exports policy to the folder named "export".


### Real-world deployment  ###
1. connect to go1
`sh {REPO_PATH}/go1_gym_deploy/scripts/go1_connect.sh`
2. ssh to go1 system
`ssh unitree@192.168.123.15`
3. run position controller node
`sudo ~/go1_gym/go1_gym_deploy/unitree_legged_sdk_bin/lcm_position`
4. start docker node
`cd ~/go1_gym/go1_gym_deploy/docker && sudo make autostart`
5. run deploy script
`python go1_gym_deploy/scripts/deploy_policy.py`

### License ###
This repository and its code are referred from
1. https://github.com/Alescontrela/AMP_for_hardware
2. https://github.com/Improbable-AI/walk-these-ways
3. https://github.com/Denys88/rl_games
