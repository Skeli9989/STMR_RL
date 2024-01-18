CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task go1base_NMR_hopturn --headless   &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task go1base_STMR_hopturn --headless  &
CUDA_VISIBLE_DEVICES=1 python3 legged_gym/scripts/evaluate.py --task go1base_TO_hopturn --headless    &
wait

