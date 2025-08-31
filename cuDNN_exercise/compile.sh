#module load CUDA/12
#module load cuDNN/9.5.0.50-CUDA-12

nvcc -I/p/software/default/stages/2025/software/cuDNN/9.5.0.50-CUDA-12/include \
    -L/p/software/default/stages/2025/software/cuDNN/9.5.0.50-CUDA-12/lib \
    -lcudnn -lcublas -o attention attention.cu