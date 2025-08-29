/**************************************************************
This code is a part of a course on cuda taught by the author: 
Lokman A. Abbas-Turki

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NB 2
#define NTPB 64

__device__ float Glob[7*NB*NTPB];	// Global variable solution

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

__global__ void biggest_k(int *In, int *Out, int N){

    __shared__ int InSh[NTPB];
    // load into shared
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        InSh[threadIdx.x] = In[idx];
    }
    __syncthreads();

    // reduction
    for (int i = blockDim.x/2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            InSh[threadIdx.x] = max(InSh[threadIdx.x], InSh[threadIdx.x+i]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        //printf("InSh[0]=%d", InSh[0]);
        Out[blockIdx.x] = InSh[0];
    }
    if (threadIdx.x == 0) {
        atomicMax(Out, InSh[0]);  
    }
}

// Has to be defined in the compilation in order to get the correct value of the macros
// __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


int main (void){

	int N = 256;
    int *h_In = (int*) malloc(N * sizeof(int));
    // Initialize array
    for(int i=0; i<N; i++){
        h_In[i]=i;
    }
    // Move array to GPU global memory
    int *d_In;
    cudaMalloc((void**)&d_In, N*sizeof(int));
    cudaMemcpy(d_In, h_In, N*sizeof(int), cudaMemcpyHostToDevice);

    int h_Out;
    int *d_Out;
    cudaMalloc((void**)&d_Out, 2*sizeof(int));

    // Launch kernel
    biggest_k<<<NB,NTPB>>>(d_In, d_Out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_Out, d_Out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Greatest value: %d", h_Out);

    free(h_In);
    cudaFree(d_In);

	return 0;
}