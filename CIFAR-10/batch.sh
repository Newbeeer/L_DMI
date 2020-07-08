#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="cifar-training"
#SBATCH --output=sample-0.4-uniform.out

# only use the following if you want email notification
#SBATCH --mail-user=xuyilun@pku.edu.cn
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

python3 CE.py --r 0.4 --s 1234 --device 0 --root '/sailhome/xuyilun/cifar'

echo 'DONE'
