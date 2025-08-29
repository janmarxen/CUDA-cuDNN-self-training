/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>

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

__global__ void empty_k(void) {}

__global__ void print_k(void) {

    // Question 1

    //if(blockIdx.x!=0)
    //	printf("Hello World!\n");

    // Compute global thread index
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

    // Compute warp index inside the block
    int WarpIdx = threadIdx.x / 32;

    // Compute global warp index
    int globalWarpIdx = globalIdx / 32;

    // Print all info in one line
    printf("Block %d, Thread %d => globalIdx: %d, WarpIdx: %d, globalWarpIdx: %d\n",
           blockIdx.x, threadIdx.x, globalIdx, WarpIdx, globalWarpIdx);

}

int main(void) {

	empty_k <<<1, 1>>> ();

	int device;
    cudaGetDevice(&device);

    // a)
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

    printf("Compute capability: %d.%d\n", major, minor);
    // b)
    size_t limit;
    cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
    printf("FIFO buffer size: %zu bytes\n", limit);

    print_k<<<64, 64>>>();
    cudaDeviceSynchronize(); 


	return 0;
}