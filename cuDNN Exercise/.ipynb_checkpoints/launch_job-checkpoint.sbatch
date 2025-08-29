#!/bin/bash
#SBATCH --job-name=cuda_programming
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=training2529
#SBATCH --partition=gpus
#SBATCH --time=00:10:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

export LD_LIBRARY_PATH=/p/software/default/stages/2025/software/cuDNN/9.5.0.50-CUDA-12/lib:$LD_LIBRARY_PATH

#./attention


nvprof ./attention