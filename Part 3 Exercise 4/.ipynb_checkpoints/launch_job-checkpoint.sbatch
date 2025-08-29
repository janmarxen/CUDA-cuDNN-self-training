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

#./batch_reduction

./batch_reduction_multiple

#!nvprof ./batch_reduction