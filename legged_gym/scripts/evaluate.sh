CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task go1base_STMR_trot0 --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task  go1base_NMR_trot0 --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task  go1base_AMP_trot0 --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task   go1base_TO_trot0 --headless --seed 1 &
wait

CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task go1base_STMR_trot1 --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task  go1base_NMR_trot1 --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task  go1base_AMP_trot1 --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task   go1base_TO_trot1 --headless --seed 1 &
wait

CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task go1base_STMR_hopturn --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task  go1base_NMR_hopturn --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task  go1base_AMP_hopturn --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task   go1base_TO_hopturn --headless --seed 1 &
wait

CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task go1base_STMR_pace0 --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task  go1base_NMR_pace0 --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task  go1base_AMP_pace0 --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task   go1base_TO_pace0 --headless --seed 1 &
wait

CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task go1base_STMR_pace1 --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task  go1base_NMR_pace1 --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task  go1base_AMP_pace1 --headless --seed 1 &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task   go1base_TO_pace1 --headless --seed 1 &
wait

