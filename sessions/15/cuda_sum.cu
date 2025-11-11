#include <cuda_runtime.h>
#include <iostream>

__global__ void prod(const float* a, const float* b, float* c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // overhead

    if (idx < N) { // overhead check out of bounds
        c[idx] = a[idx] * b[idx]; // divergence: every thread out of bounds does nothing
    }
}


// dot product: multiply a and b and return the scalar dot product
// each thread is jumping in memory:
// this is optimal because CUDA has all threads reading sequentially in groups of 32 (warps)
// UNLESS the jump is too big???
// a[0]*b[0] + a[1024]*b[1024] + a[2048]*b[2048] + ...
// a[1]*b[1] + a[1025]*b[1025] + a[2049]*b[2049] + ...
__global__ float dot(const float* a, const float* b, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // each thread is computing a partial sum
    float result = 0.0f;
    for (int i = idx; i < N; i += blockDim.x) {
        result += a[i] * b[i];
    }
    return result;
}
}




int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << prop.major << "." << prop.minor << std::endl;

    const int N = 1 << 20;
    size_t size = N * sizeof(float);
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice); // copy TO CUDA
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // synchronous: calls __syncthreads() internally

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize; // 2^20 / 256 = 2^12 = 4096
    prod<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost); // copy the memory back TO CPU

    for (int i = 0; i < N; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}