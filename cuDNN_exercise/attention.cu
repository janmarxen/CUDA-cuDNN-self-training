#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

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

void attention(
    cudnnHandle_t cudnn,
    cublasHandle_t cublas,
    float* d_X,          // input: (batch * seq_len, hidden_dim)
    float* d_Wq,         // weights: (hidden_dim, hidden_dim)
    float* d_Wk,
    float* d_Wv,
    float* d_output,     // attention output: (batch * seq_len, hidden_dim)
    int batch, int seq_len, int hidden_dim
) {
    int M = batch * seq_len;  // number of rows in X
    int K = hidden_dim;       // feature dimension
    int N = hidden_dim;       // output feature dimension

    // Allocate Q, K, V: each (batch * seq_len, hidden_dim)
    float *d_Q, *d_K, *d_V;
    cudaMalloc(&d_Q, M*N*sizeof(float)); // Q: (batch*seq_len, hidden_dim)
    cudaMalloc(&d_K, M*N*sizeof(float)); // K: (batch*seq_len, hidden_dim)
    cudaMalloc(&d_V, M*N*sizeof(float)); // V: (batch*seq_len, hidden_dim)

    // Allocate attention scores: (batch*seq_len, batch*seq_len)
    float *d_scores;
    cudaMalloc(&d_scores, M*M*sizeof(float)); // scores: (batch*seq_len, batch*seq_len)

    // Allocate softmax output: same shape as scores
    float *d_softmax;
    cudaMalloc(&d_softmax, M*M*sizeof(float)); // softmax: (batch*seq_len, batch*seq_len)

    // 1) Linear projections
    linear_projection(cublas, d_X, d_Wq, d_Q, M, N, K);
    linear_projection(cublas, d_X, d_Wk, d_K, M, N, K);
    linear_projection(cublas, d_X, d_Wv, d_V, M, N, K);

    // 2) Compute attention scores: Q * K^T -> (batch*seq_len, batch*seq_len)
    compute_attention_scores(cublas, d_Q, d_K, d_scores, M, K);

    // 3) Scale scores by 1/sqrt(hidden_dim)
    scale_scores(d_scores, M, M, hidden_dim);

    // 4) Apply softmax along sequence dimension for each batch
    softmax_forward(cudnn, d_scores, d_softmax, batch, seq_len);

    // 5) Compute attention output: softmax * V → (batch*seq_len, hidden_dim)
    attention_output(cublas, d_softmax, d_V, d_output, batch, seq_len, hidden_dim);

    // Free temporaries
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_softmax);
}


int main() {
    // Parameters
    int batch = 64;
    int seq_len = 20;
    int hidden_dim = 128;

    int M = batch * seq_len;
    int K = hidden_dim;

    // Host input
    std::vector<float> h_X(M * K);
    std::vector<float> h_Wq(K * K), h_Wk(K * K), h_Wv(K * K);
    std::vector<float> h_output(M * K, 0.0f);

    // Random initialization
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : h_X) x = dist(rng);
    for (auto& w : h_Wq) w = dist(rng);
    for (auto& w : h_Wk) w = dist(rng);
    for (auto& w : h_Wv) w = dist(rng);

    // Device memory
    float *d_X, *d_Wq, *d_Wk, *d_Wv, *d_output;
    cudaMalloc(&d_X, M*K*sizeof(float));
    cudaMalloc(&d_Wq, K*K*sizeof(float));
    cudaMalloc(&d_Wk, K*K*sizeof(float));
    cudaMalloc(&d_Wv, K*K*sizeof(float));
    cudaMalloc(&d_output, M*K*sizeof(float));

    cudaMemcpy(d_X, h_X.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wq, h_Wq.data(), K*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wk, h_Wk.data(), K*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wv, h_Wv.data(), K*K*sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS and cuDNN handles
    cublasHandle_t cublas;
    cudnnHandle_t cudnn;
    cublasCreate(&cublas);
    cudnnCreate(&cudnn);

    // Call attention
    attention(cudnn, cublas, d_X, d_Wq, d_Wk, d_Wv, d_output, batch, seq_len, hidden_dim);

    // Copy output back
    cudaMemcpy(h_output.data(), d_output, M*K*sizeof(float), cudaMemcpyDeviceToHost);

    // Print input and output
    print3DTensor(h_X, batch, seq_len, hidden_dim, "Input X");
    printMatrix(h_Wq, hidden_dim, hidden_dim, "Weight Wq");
    printMatrix(h_Wk, hidden_dim, hidden_dim, "Weight Wk");
    printMatrix(h_Wv, hidden_dim, hidden_dim, "Weight Wv");
    print3DTensor(h_output, batch, seq_len, hidden_dim, "Attention Output");

    // Free memory and handles
    cudaFree(d_X);
    cudaFree(d_Wq);
    cudaFree(d_Wk);
    cudaFree(d_Wv);
    cudaFree(d_output);
    cublasDestroy(cublas);
    cudnnDestroy(cudnn);

    return 0;
}
































