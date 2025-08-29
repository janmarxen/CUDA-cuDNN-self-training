/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include "timer.h"

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


void addVect(int *a, int *b, int *c, int length){

	int i;

	for(i=0; i<length; i++){
		c[i] = a[i] + b[i];
	}
}

__global__ void addVect_k(int *d_a, int *d_b, int *d_c, int length){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    while(idx<length)
        d_c[idx] = d_a[idx]+d_b[idx];
        idx += blockDim.x*gridDim.x;
    
}

__global__ void initVect_k(int *d_a, int *d_b, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        d_a[idx] = idx;
		d_b[idx] = 9*idx;
    }
}


int main (void){

	// Variables definition
	int *a, *b, *c;
	int i;
	
	// Length for the size of arrays
	int length = 1e9;

	Timer Tim;							// CPU timer instructions

	// Memory allocation of arrays 
	a = (int*)malloc(length*sizeof(int));
	b = (int*)malloc(length*sizeof(int));
	c = (int*)malloc(length*sizeof(int));

	// Values initialization
	for(i=0; i<length; i++){
		a[i] = i;
		b[i] = 9*i;
	}

	Tim.start();						// CPU timer instructions

	// Executing the addition 
	addVect(a, b, c, length);

	Tim.add();							// CPU timer instructions

	// Displaying the results to check the correctness 
	for(i=length-50; i<length-45; i++){
		printf(" ( %i ): %i\n", a[i]+b[i], c[i]);
	}

	printf("CPU Timer for the addition on the CPU of vectors: %f s\n", 
		   (float)Tim.getsum());		// CPU timer instructions

    // GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int *d_a;
    int *d_b; 
    int *d_c; 

    //cudaMalloc((void**)&d_a, sizeof(int)*length);
	//cudaMalloc((void**)&d_b, sizeof(int)*length);
    //cudaMalloc((void**)&d_c, sizeof(int)*length);
    cudaMallocManaged((void**)&d_a, sizeof(int)*length);
    cudaMallocManaged((void**)&d_b, sizeof(int)*length);
    cudaMallocManaged((void**)&d_c, sizeof(int)*length);

    //cudaMemcpy(a, d_a, length * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(b, d_b, length * sizeof(int), cudaMemcpyHostToDevice);
    initVect_k<<<128,128>>>(d_a, d_b, length);
    
    addVect_k<<<128,128>>>(d_a, d_b, d_c, length);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Convert to seconds
    float seconds = ms / 1000.0f;
    printf("Kernel execution time: %f seconds\n", seconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Freeing the memory
	free(a);
	free(b);
	free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


	return 0;
}