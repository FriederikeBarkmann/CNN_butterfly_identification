#!/bin/bash

##SBATCH --test-only  # Only check slurm script
#SBATCH -J bfly_test  # Job name

#SBATCH -D /leonardo_work/EUHPC_D12_020/fbarkman
#SBATCH -o /leonardo/home/userexternal/fbarkman/slurm_out/%A_%a_%x_%j.out
#SBATCH -e /leonardo/home/userexternal/fbarkman/slurm_out/%A_%a_%x_%j.err

#SBATCH --account=euhpc_d12_020
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=friederike.barkmann@uibk.ac.at

#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal  #boost_qos_dbg

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1  # only 1 per node task for torchrun
#SBATCH --cpus-per-task=32  # max 32 per node
##SBATCH --gres=gpu:4  # max 4 GPUs per node
#SBATCH --gpus-per-task=4  # GPUs per task/node
##SBATCH --cpus-per-gpu=8  # 32/4=8 cores per GPU

##SBATCH --mem=30GB  # Memory per node in K/M/G/T out of 481GB
##SBATCH --mem-per-gpu=16G

#SBATCH --time=0-20:00:00  # Run time limit

#SBATCH --array=2,4

# Using srun with multithreading
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/$SLURM_GPUS_PER_TASK*2))

srun run.sh

