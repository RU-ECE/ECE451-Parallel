#include "matrix.hpp"

void copy_to_device(uint32_t n, double*d_a, double*d_b, double*d_c, 
    matrix& a, matrix& b, matrix& c) {
        uint32_t size = n * n * sizeof(double);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), size, cudaMemcpyHostToDevice);
}

// this is hideous. RACE CONDITION (no Windsurf, it DOES NOT WORK) SLOW
__global__ void mult_cuda(uint32_t n, const double* a, const double* b, double* c) {
  uint32_t c = threadIdx.x;
  uint32_t r = threadIdx.y;
  c[r * n + c] += a[r * n + c] * b[r * n + c];
}

int main() {
  matrix a(n,n), b(n,n), c(n,n);
  copy_to_device(a, b, c);
  mult_cuda(a, b, c);
}