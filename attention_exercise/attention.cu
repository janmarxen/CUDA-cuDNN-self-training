#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

inline void __checkCUDA(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s (code %d) in %s at line %d\n",
               cudaGetErrorString(error), error, file, line);
        exit(EXIT_FAILURE);
    }
}

#define testCUDA(error) (__checkCUDA((error), __FILE__, __LINE__))

// Function to print a matrix with nice formatting
void printMatrix(const std::vector<float>& matrix, int rows, int cols, const std::string& name) {
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):\n";
    std::cout << std::string(name.length() + 15, '=') << std::endl;
    
    for (int i = 0; i < rows; i++) {
        std::cout << "[ ";
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(8) << std::setprecision(4) << std::fixed 
                      << matrix[i * cols + j];
            if (j < cols - 1) std::cout << ", ";
        }
        std::cout << " ]" << std::endl;
    }
    std::cout << std::endl;
}

// Function to print a 3D tensor (batch, seq_len, hidden_dim)
void print3DTensor(const std::vector<float>& tensor, int batch, int seq_len, int hidden_dim, const std::string& name) {
    std::cout << "\n" << name << " (" << batch << "x" << seq_len << "x" << hidden_dim << "):\n";
    std::cout << std::string(name.length() + 20, '=') << std::endl;
    
    for (int b = 0; b < batch; b++) {
        std::cout << "Batch " << b << ":\n";
        for (int s = 0; s < seq_len; s++) {
            std::cout << "  Seq " << s << ": [ ";
            for (int h = 0; h < hidden_dim; h++) {
                int idx = b * seq_len * hidden_dim + s * hidden_dim + h;
                std::cout << std::setw(6) << std::setprecision(3) << std::fixed 
                          << tensor[idx];
                if (h < hidden_dim - 1) std::cout << ", ";
            }
            std::cout << " ]" << std::endl;
        }
        if (b < batch - 1) std::cout << std::endl;
    }
    std::cout << std::endl;
}

void linear_projection(
    cublasHandle_t handle,
    float* d_X,      // input (M x K)
    float* d_W,      // weight (K x N)
    float* d_out,    // output (M x N)
    int M, int N, int K
) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_X, M,
        d_W, K,
        &beta,
        d_out, M
    );
}

void compute_attention_scores(
    cublasHandle_t handle,
    float* d_Q, float* d_K,
    float* d_scores,
    int M, int K
) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,   // Q * K^T
        M, M, K,
        &alpha,
        d_Q, M,
        d_K, M,
        &beta,
        d_scores, M
    );
}

__global__ void scale_kernel(float* data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] *= scale;
}

void scale_scores(float* d_scores, int M, int N, int hidden_dim) {
    int size = M*N;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    float scale = 1.0f / sqrtf((float)hidden_dim);
    scale_kernel<<<blocks, threads>>>(d_scores, size, scale);
    cudaDeviceSynchronize();
}

void softmax_forward(
    cudnnHandle_t cudnn,
    float* d_scores,
    float* d_softmax,
    int batch, int seq_len
) {
    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    int dims[3] = {batch, seq_len, seq_len};
    int strides[3] = {seq_len*seq_len, seq_len, 1};
    cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 3, dims, strides);

    float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(
        cudnn,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        desc, d_scores,
        &beta,
        desc, d_softmax
    );

    cudnnDestroyTensorDescriptor(desc);
}

void attention_output(
    cublasHandle_t handle,
    float* d_softmax,
    float* d_V,
    float* d_output,
    int batch, int seq_len, int hidden_dim
) {
    float alpha = 1.0f, beta = 0.0f;
    int M = hidden_dim;
    int N = seq_len;
    int K = seq_len;

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_V, M, seq_len * hidden_dim,
        d_softmax, K, seq_len * seq_len,
        &beta,
        d_output, M, seq_len * hidden_dim,
        batch
    );
}

struct AttentionBuffers {
    float *d_Q, *d_K, *d_V;        // Q, K, V: (batch*seq_len, hidden_dim)
    float *d_scores, *d_softmax;   // Attention scores and softmax: (batch*seq_len, batch*seq_len)
};

// Allocate all temporaries once
void allocate_attention_buffers(AttentionBuffers &buf, int M, int hidden_dim) {
    int K = hidden_dim;

    // Q, K, V matrices
    // Shape: (batch*seq_len, hidden_dim) = (M, K)
    testCUDA(cudaMalloc(&buf.d_Q, M*K*sizeof(float)));
    testCUDA(cudaMalloc(&buf.d_K, M*K*sizeof(float)));
    testCUDA(cudaMalloc(&buf.d_V, M*K*sizeof(float)));

    // Attention scores: Q * K^T
    // Shape: (batch*seq_len, batch*seq_len) = (M, M)
    // ⚠️ Memory bottleneck: grows quadratically with seq_len
    testCUDA(cudaMalloc(&buf.d_scores, M*M*sizeof(float)));

    // Softmax output, same shape as scores
    testCUDA(cudaMalloc(&buf.d_softmax, M*M*sizeof(float)));
}

// Free all temporaries
void free_attention_buffers(AttentionBuffers &buf) {
    testCUDA(cudaFree(buf.d_Q));
    testCUDA(cudaFree(buf.d_K));
    testCUDA(cudaFree(buf.d_V));
    testCUDA(cudaFree(buf.d_scores));
    testCUDA(cudaFree(buf.d_softmax));
}

// Attention forward pass
void attention(
    cudnnHandle_t cudnn,
    cublasHandle_t cublas,
    float* d_X,           // Input: (batch*seq_len, hidden_dim) = (M, K)
    float* d_Wq,          // Weights: (hidden_dim, hidden_dim) = (K, K)
    float* d_Wk,
    float* d_Wv,
    float* d_output,      // Output: (batch*seq_len, hidden_dim) = (M, K)
    int batch, int seq_len, int hidden_dim,
    AttentionBuffers &buf // Preallocated temporaries
) {
    int M = batch * seq_len;  // total sequence elements
    int K = hidden_dim;       // hidden dimension
    int N = hidden_dim;       // usually same as K

    // 1) Linear projections
    // Q = X * Wq
    // K = X * Wk
    // V = X * Wv
    // Shapes: (M, K) = (batch*seq_len, hidden_dim)
    // Bottleneck: large GEMMs (linear projection), but smaller than Q*K^T
    linear_projection(cublas, d_X, d_Wq, buf.d_Q, M, N, K);
    linear_projection(cublas, d_X, d_Wk, buf.d_K, M, N, K);
    linear_projection(cublas, d_X, d_Wv, buf.d_V, M, N, K);

    // 2) Compute attention scores
    // scores = Q * K^T
    // Shape: (M, M) = (batch*seq_len, batch*seq_len)
    // Main bottleneck: both in FLOPs (O(M²*K)) and memory (M*M floats). 
    // This will grow quadratically with seq_len -> why vanilla attention is infeasible for long sequences.
    compute_attention_scores(cublas, buf.d_Q, buf.d_K, buf.d_scores, M, K);

    // 3) Scale scores by 1/sqrt(hidden_dim)
    // Memory-bound: reads/writes full (M, M) matrix
    scale_scores(buf.d_scores, M, M, hidden_dim);

    // 4) Softmax along sequence dimension
    // Memory-bound, usually cheap compared to GEMM
    softmax_forward(cudnn, buf.d_scores, buf.d_softmax, batch, seq_len);

    // 5) Attention output
    // output = softmax * V
    // Shapes: (M, M) @ (M, K) -> (M, K)
    // Bottleneck: GEMM, but smaller than Q*K^T
    attention_output(cublas, buf.d_softmax, buf.d_V, d_output, batch, seq_len, hidden_dim);
}


int main() {
    int batch = 64;
    int seq_len = 512;
    int hidden_dim = 1024;

    int M = batch * seq_len;
    int K = hidden_dim;

    // Allocate input and weight matrices
    float *d_X, *d_Wq, *d_Wk, *d_Wv, *d_output;
    testCUDA(cudaMalloc(&d_X, M*K*sizeof(float)));
    testCUDA(cudaMalloc(&d_Wq, K*K*sizeof(float)));
    testCUDA(cudaMalloc(&d_Wk, K*K*sizeof(float)));
    testCUDA(cudaMalloc(&d_Wv, K*K*sizeof(float)));
    testCUDA(cudaMalloc(&d_output, M*K*sizeof(float)));

    // Initialize random values
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniform(gen, d_X, M*K);
    curandGenerateUniform(gen, d_Wq, K*K);
    curandGenerateUniform(gen, d_Wk, K*K);
    curandGenerateUniform(gen, d_Wv, K*K);
    cudaMemset(d_output, 0, M*K*sizeof(float));

    cudaDeviceSynchronize();

    // cuBLAS and cuDNN
    cublasHandle_t cublas;
    cudnnHandle_t cudnn;
    cublasCreate(&cublas);
    cudnnCreate(&cudnn);

    // Allocate attention temporaries **once**
    AttentionBuffers buf;
    allocate_attention_buffers(buf, M, hidden_dim);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    attention(cudnn, cublas, d_X, d_Wq, d_Wk, d_Wv, d_output,
              batch, seq_len, hidden_dim, buf);
    cudaDeviceSynchronize();

    // Timed run
    cudaProfilerStart();
    cudaEventRecord(start);
    attention(cudnn, cublas, d_X, d_Wq, d_Wk, d_Wv, d_output,
              batch, seq_len, hidden_dim, buf);
    cudaEventRecord(stop);
    cudaProfilerStop();
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Pure attention computation time: %.3f ms\n", milliseconds);

    // Cleanup
    free_attention_buffers(buf);

    cudaFree(d_X);
    cudaFree(d_Wq);
    cudaFree(d_Wk);
    cudaFree(d_Wv);
    cudaFree(d_output);

    curandDestroyGenerator(gen);
    cublasDestroy(cublas);
    cudnnDestroy(cudnn);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

































