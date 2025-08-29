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

// Original kernel: batched tiled matmul with fixed TILE_DIM
__global__ void multiBatch_4_k(float *A, float *B, float *C, int d) {
    int batch = blockIdx.z;
    int batch_offset = batch * d * d;
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

        As[threadIdx.x * TILE_DIM + threadIdx.y] = (aRow < d && aCol < d) ? A[batch_offset + aRow*d + aCol] : 0.0f;
        Bs[threadIdx.x * TILE_DIM + threadIdx.y] = (bRow < d && bCol < d) ? B[batch_offset + bRow*d + bCol] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++) {
            sum += As[threadIdx.x * TILE_DIM + k] * Bs[k * TILE_DIM + threadIdx.y];
        }

        __syncthreads();
    }

    if (row < d && col < d) {
        C[batch_offset + row*d + col] = sum;
    }
}

// NEW: Parameterized kernel for different tile sizes
template<int TILE>
__global__ void multiBatch_param_k(float *A, float *B, float *C, int d) {
    int batch = blockIdx.z;
    int batch_offset = batch * d * d;
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

        As[threadIdx.x * TILE + threadIdx.y] = (aRow < d && aCol < d) ? A[batch_offset + aRow*d + aCol] : 0.0f;
        Bs[threadIdx.x * TILE + threadIdx.y] = (bRow < d && bCol < d) ? B[batch_offset + bRow*d + bCol] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.x * TILE + k] * Bs[k * TILE + threadIdx.y];
        }

        __syncthreads();
    }

    if (row < d && col < d) {
        C[batch_offset + row*d + col] = sum;
    }
}

// Timing utility for original kernel
float timeKernel(dim3 grid, dim3 block, float *d_A, float *d_B, float *d_C, int d, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_C, 0, N * d * d * sizeof(float));
    cudaEventRecord(start);
    
    multiBatch_4_k<<<grid, block, 2*TILE_DIM*TILE_DIM*sizeof(float)>>>(d_A, d_B, d_C, d);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// NEW: Template-based timing utility for parameterized kernels
template<int TILE>
float timeKernelParam(dim3 grid, dim3 block, float *d_A, float *d_B, float *d_C, int d, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_C, 0, N * d * d * sizeof(float));
    cudaEventRecord(start);
    
    multiBatch_param_k<TILE><<<grid, block, 2*TILE*TILE*sizeof(float)>>>(d_A, d_B, d_C, d);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// Weak scaling: increase d proportionally to N
void weakScaling(int N_start, int N_end, int d_start, const char* file_name) {
    FILE *f = fopen(file_name, "w");
    fprintf(f, "N,d,time_ms\n");

    for (int N = N_start; N <= N_end; N *= 2) {
        int d = d_start * N; // weak scaling: problem size grows with N
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N * d * d * sizeof(float));
        cudaMalloc(&d_B, N * d * d * sizeof(float));
        cudaMalloc(&d_C, N * d * d * sizeof(float));

        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, d_A, N*d*d);
        curandGenerateUniform(gen, d_B, N*d*d);

        dim3 threads(TILE_DIM, TILE_DIM);
        dim3 grid((d + TILE_DIM - 1)/TILE_DIM, (d + TILE_DIM - 1)/TILE_DIM, N);

        float t = timeKernel(grid, threads, d_A, d_B, d_C, d, N);
        fprintf(f, "%d,%d,%f\n", N, d, t);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        curandDestroyGenerator(gen);
    }

    fclose(f);
}

void strongScaling(int N, int d, const char* file_name) {
    FILE *f = fopen(file_name, "w");
    fprintf(f, "threadsPerBlock,time_ms\n");

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*d*d*sizeof(float));
    cudaMalloc(&d_B, N*d*d*sizeof(float));
    cudaMalloc(&d_C, N*d*d*sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, d_A, N*d*d);
    curandGenerateUniform(gen, d_B, N*d*d);

    // Test different tile sizes with matching kernels
    struct {int tile; const char* name;} configs[] = {{4, "4x4"}, {8, "8x8"}, {16, "16x16"}, {32, "32x32"}};
    
    for (int i = 0; i < 4; i++) {
        int TILE = configs[i].tile;
        dim3 threads(TILE, TILE);
        dim3 grid((d + TILE - 1)/TILE, (d + TILE - 1)/TILE, N);
        
        float t;
        switch(TILE) {
            case 4:  t = timeKernelParam<4>(grid, threads, d_A, d_B, d_C, d, N); break;
            case 8:  t = timeKernelParam<8>(grid, threads, d_A, d_B, d_C, d, N); break;
            case 16: t = timeKernelParam<16>(grid, threads, d_A, d_B, d_C, d, N); break;
            case 32: t = timeKernelParam<32>(grid, threads, d_A, d_B, d_C, d, N); break;
            default: t = -1; break;
        }
        
        if (t >= 0) {
            fprintf(f, "%d,%f\n", TILE*TILE, t);
            printf("Tile %dx%d (%d threads): %.6f ms\n", TILE, TILE, TILE*TILE, t);
        }
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    curandDestroyGenerator(gen);
    fclose(f);
}

int main() {
    weakScaling(1, 16, 128, "weak_scaling.csv");
    strongScaling(16, 1024, "strong_scaling.csv");
    return 0;
}