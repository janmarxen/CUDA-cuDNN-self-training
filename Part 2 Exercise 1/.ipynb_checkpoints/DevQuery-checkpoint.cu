/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

int main (void){

	int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("Number of CUDA devices: %d\n\n", deviceCount);

    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    // a)
    printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    // b)
    printf("  Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    //  
    // c) 
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    // 
    // d) 
    printf("  Warp size: %d\n", prop.warpSize);
    // e)
    printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    // f)
    printf("  Registers per block: %d\n", prop.regsPerBlock);
    // h)
    int sm_count = prop.multiProcessorCount;
    int cuda_cores_per_sm;
    
    // Determine CUDA cores per SM based on compute capability
    if (prop.major == 8) {          // Ampere (RTX 30 series)
        cuda_cores_per_sm = 128;
    } else if (prop.major == 7) {    // Turing (RTX 20 series)
        cuda_cores_per_sm = 64;
    } else if (prop.major == 6) {    // Pascal (GTX 10 series)
        cuda_cores_per_sm = 128;
    } else if (prop.major == 5) {    // Maxwell
        cuda_cores_per_sm = 128;
    }
    // Add more architectures as needed
    
    int total_cuda_cores = sm_count * cuda_cores_per_sm;
    printf("CUDA cores: %d\n", total_cuda_cores);

	return 0;
}