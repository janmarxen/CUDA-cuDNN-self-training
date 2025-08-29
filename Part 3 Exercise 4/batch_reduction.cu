/**************************************************************
This code is a part of a course on cuda taught by the author: 
Lokman A. Abbas-Turki

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

__global__  void prod_k1(float *a, float *b, float *c, int n){
    __shared__ float local_c[2];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        //printf("threadIdx.x=%d, idx=%d, a[idx]*b[idx]=%f\n", threadIdx.x, idx, a[idx]*b[idx]);
        local_c[threadIdx.x] = a[idx]*b[idx];
    }
    __syncthreads();
    if (threadIdx.x==0) {
        local_c[0] += local_c[1];
        atomicAdd(c, local_c[0]);
    }

}

__global__  void prod_k2(float *a, float *b, float *c, int n){
    __shared__ float local_c[256];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Multiplication
    if (idx < n) {
        //printf("threadIdx.x=%d, idx=%d, a[idx]*b[idx]=%f\n", threadIdx.x, idx, a[idx]*b[idx]);
        local_c[threadIdx.x] = a[idx]*b[idx];
    }
    __syncthreads();

    // Sum reduction
    for (int i = blockDim.x/2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            local_c[threadIdx.x] = local_c[threadIdx.x] + local_c[threadIdx.x+i];
        }
        __syncthreads();
    }
    
    if (threadIdx.x==0) {
        atomicAdd(c, local_c[0]);
    }

}

// Has to be defined in the compilation in order to get the correct value of the macros
// __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


int main (void){

	int n = 512;
    float *h_a = (float*) malloc(n * sizeof(float));
    float *h_b = (float*) malloc(n * sizeof(float));
    // Initialize arrays
    for(int i=0; i<n; i++){
        h_a[i]=i;
        h_b[i]=i+9;
    }
    // Move a,b arrays to GPU global memory
    float *d_a;
    float *d_b;
    cudaMalloc((void**)&d_a, n*sizeof(float));
    cudaMalloc((void**)&d_b, n*sizeof(float));
    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*sizeof(float), cudaMemcpyHostToDevice);

    float h_c = 0;
    float *d_c;
    cudaMalloc((void**)&d_c, sizeof(float));
    cudaMemset(d_c, 0, sizeof(float));

    // Ex. 1.a)
    //int nb = n/2;
    //int ntpb = 2;
    //prod_k1<<<nb,ntpb>>>(d_a, d_b, d_c, n);

    // Ex.1.b)
    int nb = n/256;
    int ntpb = 256;
    prod_k2<<<nb,ntpb>>>(d_a, d_b, d_c, n);
    
    cudaDeviceSynchronize();
    cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result value: %f\n", h_c);

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

	return 0;
}