/**************************************************************
This code is a part of a course on cuda taught by the author: 
Lokman A. Abbas-Turki

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <cuda.h>

#define TILE_DIM 16

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Host-side matrix multiplication for validation
void validateResult(float *A, float *B, float *C, int d, int N) {
    float *ref = (float*)calloc(N * d * d, sizeof(float));

    // Compute reference result on CPU
    for (int n = 0; n < N; n++) {
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                float sum = 0.0f;
                for (int k = 0; k < d; k++) {
                    sum += A[n * d * d + i * d + k] * B[n * d * d + k * d + j];
                }
                ref[n * d * d + i * d + j] = sum;
            }
        }
    }

    // Compare with GPU results
    int errors = 0;
    for (int i = 0; i < N * d * d; i++) {
        float diff = fabs(ref[i] - C[i]);
        if (diff > 1e-3) {  // tolerance
            printf("Mismatch at index %d: CPU = %f, GPU = %f, diff = %f\n",
                   i, ref[i], C[i], diff);
            errors++;
        }
    }

    if (errors == 0) {
        printf("Validation PASSED! CPU and GPU results match.\n");
    } else {
        printf("Validation FAILED: %d mismatches found.\n", errors);
    }

    free(ref);
}


__global__ void multiBatch_k(float *A, float *B, float *C, int d) {

    for (int i=0; i<d; i++) {
        for (int j=0; j<d; j++) {
            for (int k=0; k<d; k++) {
                C[blockIdx.x * d * d + i * d + j] += A[blockIdx.x * d * d + i * d + k] * B[blockIdx.x * d * d + k * d + j];
            }
        }
    }
}

__global__ void multiBatch_2_k(float *A, float *B, float *C, int d) {
    extern __shared__ float shared[];
    float *As = shared;
    float *Bs = shared + d*d;

    int n = blockIdx.x; // each block handles one matrix
    int tid = threadIdx.x; // each thread handles one C entry
    int row = tid / d;
    int col = tid % d;

    // Load A and B into shared memory
    for (int i = tid; i < d*d; i += blockDim.x) {
        As[i] = A[n*d*d + i];
        Bs[i] = B[n*d*d + i];
    }
    __syncthreads();

    // Compute C entry
    float sum = 0;
    for (int k = 0; k < d; k++) {
        sum += As[row*d + k] * Bs[k*d + col];
    }
    C[n*d*d + row*d + col] = sum;
}

__global__ void multiBatch_3_k(float *A, float *B, float *C, int d) {
    extern __shared__ float Bcol[]; 
    int n = blockIdx.x;             // one block per matrix
    float *local_A = &A[n * d * d];
    float *local_B = &B[n * d * d];
    float *local_C = &C[n * d * d];

    int row = threadIdx.x;          // each thread computes one row of C

    for (int col = 0; col < d; col++) {
         
        // Load column 'col' of B into shared memory
        Bcol[row] = local_B[row * d + col];
        __syncthreads();

        // Compute the dot product of row 'row' of A with column 'col' of B
        float sum = 0.0f;
        for (int k = 0; k < d; k++) {
            sum += local_A[row * d + k] * Bcol[k];
        }
        local_C[row * d + col] = sum;
        __syncthreads(); // ensure Bcol is not overwritten before all threads finish
    }
    
}

__global__ void multiBatch_4_k(float *A, float *B, float *C, int d) {

    // Define batch and tile indices
    int batch = blockIdx.z;
    int batch_offset = batch * d * d;

    // Define global indices
    int row = blockIdx.x*TILE_DIM + threadIdx.x;
    int col = blockIdx.y*TILE_DIM + threadIdx.y;

    __shared__ float As[TILE_DIM*TILE_DIM];
    __shared__ float Bs[TILE_DIM*TILE_DIM];

    float sum = 0.0f;

    int num_tiles_k = (d + TILE_DIM - 1) / TILE_DIM;
    
    for (int tile_k = 0; tile_k < num_tiles_k; tile_k++) {
    
        // Compute global indices for this tile
        int aRow = row;
        int aCol = tile_k * TILE_DIM + threadIdx.y;
        int bRow = tile_k * TILE_DIM + threadIdx.x;
        int bCol = col;
    
        // Load elements into shared memory
        As[threadIdx.x * TILE_DIM + threadIdx.y] = A[batch_offset + aRow*d + aCol];
        Bs[threadIdx.x * TILE_DIM + threadIdx.y] = B[batch_offset + bRow*d + bCol];
    
        __syncthreads();
    
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_DIM; k++) {
            sum += As[threadIdx.x * TILE_DIM + k] * Bs[k * TILE_DIM + threadIdx.y];
        }
    
        __syncthreads(); // ensure all threads are done before loading next tile
    }
    
    // Write final value to C
    if (row < d && col < d) {
        C[batch_offset + row*d + col] = sum;
    }

}



// Has to be defined in the compilation in order to get the correct value of the macros
// __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


int main (void){

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    int N = 8;
    int d = 128;
    float *d_A;
    float *d_B;
    cudaMalloc((void**)&d_A, N * d * d * sizeof(float));
    cudaMalloc((void**)&d_B, N * d * d * sizeof(float));

    curandGenerateUniform(gen, d_A, N * d * d);
    curandGenerateUniform(gen, d_B, N * d * d);

    printf("Initialized A, B arrays on device.\n");

    float *h_C = (float*)calloc(N * d * d, sizeof(float));  
    float *d_C;
    cudaMalloc((void**)&d_C, N * d * d * sizeof(float));
    cudaMemset(d_C, 0, N * d * d * sizeof(float));

    printf("Initialized C array on host and device.\n");

    // Ex. 1)
    //int nb = N;
    //int ntpb = 1;
    //multiBatch_k<<<nb,ntpb>>>(d_A, d_B, d_C, d);

    // Ex. 2)
    //int nb = N;
    //int ntpb = d * d;
    //size_t sh_mem = 2*d*d*sizeof(float);
    //multiBatch_2_k<<<nb,ntpb,sh_mem>>>(d_A, d_B, d_C, d);

    // Ex. 3)
    //int nb = N;
    //int ntpb = d;
    //size_t sh_mem = d*sizeof(float);
    //multiBatch_3_k<<<nb,ntpb,sh_mem>>>(d_A, d_B, d_C, d);

    // Ex. 4)
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((d + TILE_DIM - 1) / TILE_DIM,  // tiles in row dimension
          (d + TILE_DIM - 1) / TILE_DIM,  // tiles in column dimension
          N);                             // number of matrices in batch
    size_t sh_mem = 2 * TILE_DIM * TILE_DIM * sizeof(float);
    multiBatch_4_k<<<grid, threads, sh_mem>>>(d_A, d_B, d_C, d);
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, N * d * d * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate
    float *h_A = (float*)malloc(N * d * d * sizeof(float));
    float *h_B = (float*)malloc(N * d * d * sizeof(float));
    cudaMemcpy(h_A, d_A, N * d * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, N * d * d * sizeof(float), cudaMemcpyDeviceToHost);
    validateResult(h_A, h_B, h_C, d, N);

    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

	return 0;
}