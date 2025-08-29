#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cuda.h>

#define TILE_DIM 16

// CUDA error check
void testCUDA(cudaError_t error, const char *file, int line)  {
    if (error != cudaSuccess) {
       printf("CUDA error in file %s at line %d: %s\n", file, line, cudaGetErrorString(error));
       exit(EXIT_FAILURE);
    } 
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

// Single matrix multiplication kernel with fixed TILE_DIM
__global__ void singleMatMul_k(float *A, float *B, float *C, int d) {
    int row = blockIdx.x * TILE_DIM + threadIdx.x;
    int col = blockIdx.y * TILE_DIM + threadIdx.y;

    __shared__ float As[TILE_DIM*TILE_DIM];
    __shared__ float Bs[TILE_DIM*TILE_DIM];

    float sum = 0.0f;
    int num_tiles_k = (d + TILE_DIM - 1) / TILE_DIM;

    for (int tile_k = 0; tile_k < num_tiles_k; tile_k++) {
        int aRow = row;
        int aCol = tile_k * TILE_DIM + threadIdx.y;
        int bRow = tile_k * TILE_DIM + threadIdx.x;
        int bCol = col;

        As[threadIdx.x * TILE_DIM + threadIdx.y] = (aRow < d && aCol < d) ? A[aRow*d + aCol] : 0.0f;
        Bs[threadIdx.x * TILE_DIM + threadIdx.y] = (bRow < d && bCol < d) ? B[bRow*d + bCol] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++) {
            sum += As[threadIdx.x * TILE_DIM + k] * Bs[k * TILE_DIM + threadIdx.y];
        }

        __syncthreads();
    }

    if (row < d && col < d) {
        C[row*d + col] = sum;
    }
}

// Parameterized kernel for different tile sizes
template<int TILE>
__global__ void singleMatMul_param_k(float *A, float *B, float *C, int d) {
    int row = blockIdx.x * TILE + threadIdx.x;
    int col = blockIdx.y * TILE + threadIdx.y;

    __shared__ float As[TILE*TILE];
    __shared__ float Bs[TILE*TILE];

    float sum = 0.0f;
    int num_tiles_k = (d + TILE - 1) / TILE;

    for (int tile_k = 0; tile_k < num_tiles_k; tile_k++) {
        int aRow = row;
        int aCol = tile_k * TILE + threadIdx.y;
        int bRow = tile_k * TILE + threadIdx.x;
        int bCol = col;

        As[threadIdx.x * TILE + threadIdx.y] = (aRow < d && aCol < d) ? A[aRow*d + aCol] : 0.0f;
        Bs[threadIdx.x * TILE + threadIdx.y] = (bRow < d && bCol < d) ? B[bRow*d + bCol] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.x * TILE + k] * Bs[k * TILE + threadIdx.y];
        }

        __syncthreads();
    }

    if (row < d && col < d) {
        C[row*d + col] = sum;
    }
}

// Timing utility for single matrix multiplication
float timeMatMul(dim3 grid, dim3 block, float *d_A, float *d_B, float *d_C, int d) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Clear output matrix
    cudaMemset(d_C, 0, d * d * sizeof(float));
    
    cudaEventRecord(start);
    singleMatMul_k<<<grid, block>>>(d_A, d_B, d_C, d);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// Template-based timing utility for parameterized kernels
template<int TILE>
float timeMatMulParam(dim3 grid, dim3 block, float *d_A, float *d_B, float *d_C, int d) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_C, 0, d * d * sizeof(float));
    
    cudaEventRecord(start);
    singleMatMul_param_k<TILE><<<grid, block>>>(d_A, d_B, d_C, d);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// Calculate GFLOPS for matrix multiplication
float calculateGFLOPS(int d, float time_ms) {
    // Matrix multiplication: d^3 multiply-add operations = 2*d^3 FLOPs
    double flops = 2.0 * (double)d * d * d;
    double time_s = time_ms / 1000.0;
    return (float)(flops / time_s / 1e9);
}

// Weak scaling: matrix size grows with number of thread blocks
void weakScaling(const char* file_name) {
    FILE *f = fopen(file_name, "w");
    fprintf(f, "matrix_size,num_blocks,time_ms,gflops\n");
    
    printf("=== WEAK SCALING ANALYSIS ===\n");
    printf("Concept: Matrix size grows proportionally to number of thread blocks\n");
    printf("Expectation: Time should remain roughly constant if scaling is ideal\n\n");

    // Start with base case: 16x16 matrix (1 block per dimension)
    int base_blocks = 1;
    int base_d = TILE_DIM * base_blocks; // 16
    
    for (int blocks_per_dim = 1; blocks_per_dim <= 64; blocks_per_dim *= 2) {
        int d = TILE_DIM * blocks_per_dim;
        int total_blocks = blocks_per_dim * blocks_per_dim;
        
        printf("Testing: %dx%d matrix (%d blocks per dim, %d total blocks)\n", 
               d, d, blocks_per_dim, total_blocks);
        
        float *d_A, *d_B, *d_C;
        testCUDA(cudaMalloc(&d_A, d * d * sizeof(float)));
        testCUDA(cudaMalloc(&d_B, d * d * sizeof(float)));
        testCUDA(cudaMalloc(&d_C, d * d * sizeof(float)));

        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, d_A, d*d);
        curandGenerateUniform(gen, d_B, d*d);

        dim3 threads(TILE_DIM, TILE_DIM);
        dim3 grid(blocks_per_dim, blocks_per_dim);

        float time_ms = timeMatMul(grid, threads, d_A, d_B, d_C, d);
        float gflops = calculateGFLOPS(d, time_ms);
        
        fprintf(f, "%d,%d,%.6f,%.2f\n", d, total_blocks, time_ms, gflops);
        printf("  Result: %.3f ms, %.1f GFLOPS\n", time_ms, gflops);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        curandDestroyGenerator(gen);
    }
    
    printf("\n");
    fclose(f);
}

// Strong scaling: fixed matrix size, vary tile size (and thus number of threads)
void strongScaling(int matrix_size, const char* file_name) {
    FILE *f = fopen(file_name, "w");
    fprintf(f, "matrix_size,tile_size,threads_per_block,total_blocks,time_ms,gflops,efficiency\n");
    
    printf("=== STRONG SCALING ANALYSIS ===\n");
    printf("Concept: Fixed matrix size (%dx%d), vary tile size to increase parallelism\n", matrix_size, matrix_size);
    printf("Expectation: Time should decrease as we use more threads (up to hardware limits)\n\n");
    
    // Pre-allocate matrices once
    float *d_A, *d_B, *d_C;
    testCUDA(cudaMalloc(&d_A, matrix_size * matrix_size * sizeof(float)));
    testCUDA(cudaMalloc(&d_B, matrix_size * matrix_size * sizeof(float)));
    testCUDA(cudaMalloc(&d_C, matrix_size * matrix_size * sizeof(float)));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, d_A, matrix_size*matrix_size);
    curandGenerateUniform(gen, d_B, matrix_size*matrix_size);
    
    float baseline_time = 0.0f;
    
    // Test different tile sizes (must divide matrix_size evenly for clean results)
    int tile_sizes[] = {4, 8, 16, 32};
    int num_tiles = sizeof(tile_sizes) / sizeof(tile_sizes[0]);
    
    for (int i = 0; i < num_tiles; i++) {
        int tile_size = tile_sizes[i];
        
        // Skip if tile size is too big for the matrix or exceeds thread limits
        if (tile_size > matrix_size || tile_size * tile_size > 1024) {
            printf("Skipping tile size %d (too large)\n", tile_size);
            continue;
        }
        
        int blocks_per_dim = (matrix_size + tile_size - 1) / tile_size;
        int total_blocks = blocks_per_dim * blocks_per_dim;
        int threads_per_block = tile_size * tile_size;
        
        printf("Testing: %dx%d tiles (%d threads/block, %d blocks per dim, %d total blocks)\n", 
               tile_size, tile_size, threads_per_block, blocks_per_dim, total_blocks);
        
        dim3 threads(tile_size, tile_size);
        dim3 grid(blocks_per_dim, blocks_per_dim);

        float time_ms;
        switch(tile_size) {
            case 4:  time_ms = timeMatMulParam<4>(grid, threads, d_A, d_B, d_C, matrix_size); break;
            case 8:  time_ms = timeMatMulParam<8>(grid, threads, d_A, d_B, d_C, matrix_size); break;
            case 16: time_ms = timeMatMulParam<16>(grid, threads, d_A, d_B, d_C, matrix_size); break;
            case 32: time_ms = timeMatMulParam<32>(grid, threads, d_A, d_B, d_C, matrix_size); break;
            default: 
                printf("Unsupported tile size %d\n", tile_size);
                continue;
        }
        float gflops = calculateGFLOPS(matrix_size, time_ms);
        
        if (i == 0) {
            baseline_time = time_ms;
        }
        
        // Efficiency: (baseline_time / current_time) / (threads_ratio) * 100%
        int baseline_threads = tile_sizes[0] * tile_sizes[0];
        float threads_ratio = (float)threads_per_block / baseline_threads;
        float efficiency = (baseline_time / time_ms) / threads_ratio * 100.0f;
        
        fprintf(f, "%d,%d,%d,%d,%.6f,%.2f,%.1f\n", 
                matrix_size, tile_size, threads_per_block, total_blocks, time_ms, gflops, efficiency);
        printf("  Result: %.3f ms, %.1f GFLOPS, %.1f%% efficiency\n", time_ms, gflops, efficiency);
    }
    
    printf("\n");
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    curandDestroyGenerator(gen);
    fclose(f);
}

int main() {
    // Query GPU properties for context
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d, Max threads per block: %d\n", prop.multiProcessorCount, prop.maxThreadsPerBlock);
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Tile size (fixed): %dx%d = %d threads per block\n\n", TILE_DIM, TILE_DIM, TILE_DIM*TILE_DIM);
    
    // Run scaling tests
    weakScaling("weak_scaling_single.csv");
    strongScaling(1024, "strong_scaling_single.csv");  
    
    printf("Analysis complete. Check CSV files for detailed results.\n");
    return 0;
}