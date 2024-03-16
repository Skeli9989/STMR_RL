#!/bin/bash

ROBOT="a1base"
declare -a SEEDS=("1" "2" "3" "4" "5")
# declare -a MOTIONS=("trot0" "trot1" "pace0" "pace1" "hopturn" "sidesteps")
declare -a MOTIONS=("hopturn" "sidesteps")

for SEED in "${SEEDS[@]}"; do
    for MOTION in "${MOTIONS[@]}"; do
        CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task "${ROBOT}_STMR_${MOTION}" --headless --seed "${SEED}" &
        CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task "${ROBOT}_NMR_${MOTION}" --headless --seed "${SEED}" &
        CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task "${ROBOT}_AMP_${MOTION}" --headless --seed "${SEED}" &
        CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task "${ROBOT}_TO_${MOTION}" --headless --seed "${SEED}" &
        wait
    done
done
