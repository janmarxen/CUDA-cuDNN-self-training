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

#./batch_mat_mul
#./batch_mat_mul_scaling
#./single_mat_mul_scaling

#nvprof ./batch_mat_mul


# List of tile sizes you want to profile
TILE_SIZES=(4 8 16 32)

# Loop over tile sizes
for TILE in "${TILE_SIZES[@]}"; do
    echo "Profiling tile size $TILE"

    # Use nvprof with a separate log file for each tile size
    nvprof --log-file nvprof_tile${TILE}.txt ./single_mat_mul_profiling $TILE
done