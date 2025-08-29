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

__global__  void prod_k3(float *a, float *b, float *c, int n){
    float local_c=0;
    for (int i=0; i<n; i++){
        local_c+=a[blockIdx.x * blockDim.x + i] + b[blockIdx.x * blockDim.x + i];
    }
    c[blockIdx.x] = local_c;
}

__global__  void prod_k4(float *a, float *b, float *c, int n){
    extern __shared__ float local_c[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < gridDim.x*n) {
        local_c[threadIdx.x] = a[idx] * b[idx];
    }
    __syncthreads();

    // Sum reduction
    for (int i = blockDim.x/2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            local_c[threadIdx.x] = local_c[threadIdx.x] + local_c[threadIdx.x+i];
        }
        __syncthreads();
    }

    // Fill result
    if (threadIdx.x==0)
        c[blockIdx.x]=local_c[0];
}

// Has to be defined in the compilation in order to get the correct value of the macros
// __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


int main (void){

	int n = 64;
    int N = 64;
    float *h_a = (float*) malloc(n * N * sizeof(float));
    float *h_b = (float*) malloc(n * N * sizeof(float));
    // Initialize arrays
    for (int i=0; i<N; i++){
        for(int j=0; j<n; j++){
            h_a[i*n+j]=j;
            h_b[i*n+j]=j+9;
        }
    }
    printf("Initialized host arrays.\n");
    // Move a,b arrays to GPU global memory
    float *d_a;
    float *d_b;
    cudaMalloc((void**)&d_a, n * N * sizeof(float));
    cudaMalloc((void**)&d_b, n * N * sizeof(float));
    cudaMemcpy(d_a, h_a, n * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * N * sizeof(float), cudaMemcpyHostToDevice);

    float *h_c = (float*)calloc(N, sizeof(float));  
    float *d_c;
    cudaMalloc((void**)&d_c, N * sizeof(float));
    cudaMemset(d_c, 0, N * sizeof(float));

    printf("Copied host arrays to device.\n");

    // Ex. 2.a)
    //int nb = N;
    //int ntpb = 1;
    //prod_k3<<<nb,ntpb>>>(d_a, d_b, d_c, n);

    // Ex. 2.b)
    int nb = N;
    int ntpb = n;
    size_t shmem_size = n * sizeof(float);
    prod_k4<<<nb,ntpb,shmem_size>>>(d_a, d_b, d_c, n);
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i<N; i++){
        printf("Result value h_c[%d]: %f\n", i, h_c[i]);
    }

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

	return 0;
}