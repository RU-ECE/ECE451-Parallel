#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void prod(const float* a, const float* b, float* c, const int N) {
	if (const auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N) // overhead check out of bounds
		c[idx] = a[idx] * b[idx]; // divergence: every thread out of bounds does nothing
}

// dot product: multiply a and b and return the scalar dot product
// each thread is jumping in memory:
// this is optimal because CUDA has all threads reading sequentially in groups of 32 (warps)
// UNLESS the jump is too big???
// a[0]*b[0] + a[1024]*b[1024] + a[2048]*b[2048] + ...
// a[1]*b[1] + a[1025]*b[1025] + a[2049]*b[2049] + ...
__global__ float dot(const float* a, const float* b, const int N) {
	const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	// each thread is computing a partial sum
	auto result = 0.0f;
	for (auto i = idx; i < N; i += blockDim.x)
		result += a[i] * b[i];
	return result;
}

int main() {
	cudaDeviceProp prop{};
	cudaGetDeviceProperties(&prop, 0);
	cout << prop.major << "." << prop.minor << endl;
	constexpr auto N = 1 << 20;
	auto size = N * sizeof(float);
	auto h_a = static_cast<float*>(malloc(size));
	auto h_b = static_cast<float*>(malloc(size));
	auto h_c = static_cast<float*>(malloc(size));
	for (auto i = 0; i < N; i++) {
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
	auto blockSize = 256;
	auto numBlocks = (N + blockSize - 1) / blockSize; // 2^20 / 256 = 2^12 = 4096
	prod<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost); // copy the memory back TO CPU
	for (auto i = 0; i < N; i++)
		cout << h_c[i] << " ";
	cout << endl;
	free(h_a);
	free(h_b);
	free(h_c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
