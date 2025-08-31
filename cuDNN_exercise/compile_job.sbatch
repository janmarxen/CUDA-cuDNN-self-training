#!/bin/bash
#SBATCH --job-name=compile_cuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=p200981
#SBATCH --qos=short
##SBATCH --partition=cpu
#SBATCH --time=00:05:00
#SBATCH --output=compile_%j.out
#SBATCH --error=compile_%j.err

# Load CUDA module (should work on compute nodes)
module load CUDA/12
module load cuDNN/9.5.0.50-CUDA-12

# Compile the attention.cu program
nvcc -I${CUDNN_ROOT}/include \
    -L${CUDNN_ROOT}/lib \
    -lcudnn -lcublas -o attention attention.cu

echo "Compilation completed!"
