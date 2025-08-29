#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 4096  // Matrix size

// CUDA error check
void testCUDA(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA error in file %s at line %d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

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

// Template-based timing utility
template<int TILE>
float timeMatMulParam(float *d_A, float *d_B, float *d_C, int d) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_C, 0, d * d * sizeof(float));

    dim3 threads(TILE, TILE);
    dim3 blocks((d + TILE - 1)/TILE, (d + TILE - 1)/TILE);

    cudaEventRecord(start);
    singleMatMul_param_k<TILE><<<blocks, threads>>>(d_A, d_B, d_C, d);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// Calculate GFLOPS
float calculateGFLOPS(int d, float time_ms) {
    double flops = 2.0 * (double)d * d * d;
    double time_s = time_ms / 1000.0;
    return (float)(flops / time_s / 1e9);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <tile_size>\n", argv[0]);
        return 1;
    }

    int tile_size = atoi(argv[1]);
    printf("Profiling matrix size %dx%d with tile size %d\n", N, N, tile_size);

    // Allocate host matrices
    float *h_A = (float*)malloc(N*N*sizeof(float));
    float *h_B = (float*)malloc(N*N*sizeof(float));
    float *h_C = (float*)malloc(N*N*sizeof(float));

    // Seed random generator
    srand((unsigned int)time(NULL));
    
    // Initialize with random floats in [0,1)
    for (int i = 0; i < N*N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device matrices
    float *d_A, *d_B, *d_C;
    testCUDA(cudaMalloc(&d_A, N*N*sizeof(float)));
    testCUDA(cudaMalloc(&d_B, N*N*sizeof(float)));
    testCUDA(cudaMalloc(&d_C, N*N*sizeof(float)));

    testCUDA(cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice));

    float time_ms = 0.0f;
    float gflops = 0.0f;

    // Launch the appropriate templated kernel
    switch(tile_size) {
        case 4:  time_ms = timeMatMulParam<4>(d_A, d_B, d_C, N); break;
        case 8:  time_ms = timeMatMulParam<8>(d_A, d_B, d_C, N); break;
        case 16: time_ms = timeMatMulParam<16>(d_A, d_B, d_C, N); break;
        case 32: time_ms = timeMatMulParam<32>(d_A, d_B, d_C, N); break;
        default:
            printf("Unsupported tile size %d\n", tile_size);
            return 1;
    }

    gflops = calculateGFLOPS(N, time_ms);

    printf("Time: %.3f ms, GFLOPS: %.2f\n", time_ms, gflops);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
