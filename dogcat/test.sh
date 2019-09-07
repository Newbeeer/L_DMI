# test - r for noise amount, s for random seed
CUDA_VISIBLE_DEVICES=0 python3 CE.py --r=0.1 --s=1234
CUDA_VISIBLE_DEVICES=0 python3 FW.py --r=0.1 --s=1234
CUDA_VISIBLE_DEVICES=0 python3 GCE.py --r=0.1 --s=1234
CUDA_VISIBLE_DEVICES=0 python3 LCCN.py --r=0.1 --s=1234
CUDA_VISIBLE_DEVICES=0 python3 DMI.py --r=0.1 --s=1234
