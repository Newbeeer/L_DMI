# test - r for noise amount, s for random seed, c for case
CUDA_VISIBLE_DEVICES=1 python3 main.py --r=0.1 --s=1234 --c=1
CUDA_VISIBLE_DEVICES=1 python3 main.py --r=0.1 --s=1234 --c=2
CUDA_VISIBLE_DEVICES=1 python3 main.py --r=0.1 --s=1234 --c=3
