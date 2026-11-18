#include "matrix.hpp"

void copy_to_device(uint32_t n, double*d_a, double*d_b, double*d_c, 
    matrix& a, matrix& b, matrix& c) {
        uint32_t size = n * n * sizeof(double);
    cudaMalloc(&d_a, size); // know this!
//ILLEGAL: we cannot access device memory from host cout << *d_a;
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), size, cudaMemcpyHostToDevice);
}

// this is hideous. RACE CONDITION (no Windsurf, it DOES NOT WORK) SLOW
__global__ void mult_cuda(uint32_t n, const double* a, const double* b, double* c) {
  uint32_t col = threadIdx.x;
  uint32_t row = threadIdx.y;
  c[row * n + col] += a[row * n + col] * b[row * n + col];
}

// also done wrong
__global__ void multi_cuda_atomic(uint32_t n, float c[], const float a[], const float b[]) {
  uint32_t col = threadIdx.x;
  uint32_t row = threadIdx.y;
  atomicAdd(&c[row * n + col], a[row * n + col] * b[row * n + col]);
}

__global__ void multi_cuda_dotproduct(uint32_t n, float c[], const float a[], const float b[]) {
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  float dot = 0; // this is in a register

  for (int i = 0; i < n; i++) {
    dot += a[row * n + i] * b[i * n + col];
  }
  c[row * n + col] = dot; // write once at the end
}

// TILE is a compile-time constant. Shared-memory array bounds must be.
// Change it and recompile to test 8, 16, 32. Block is TILE x TILE, so TILE <= 32.
//
// float temp[32][32];  inside the kernel is NOT registers and NOT shared.
//   32*32 = 1024 floats per thread. The hardware cap is 255 registers/thread,
//   and a runtime index cannot address a register. nvcc puts that array in
//   local memory (per-thread DRAM). static would only give it static lifetime,
//   still not registers.
// __shared__ is one on-chip tile per BLOCK, size fixed at compile time.
// Check with: nvcc -Xptxas -v   (lmem means it spilled out of registers)
template<int TILE>
__global__ void mult_tiled(uint32_t n, const float* a, const float* b, float* c) {
  __shared__ float As[TILE][TILE]; // this is stored in the same memory
  __shared__ float Bs[TILE][TILE]; // as L1 cache but it's locked

  uint32_t row = blockIdx.y * TILE + threadIdx.y;
  uint32_t col = blockIdx.x * TILE + threadIdx.x;
  float dot = 0; // register

  for (uint32_t t = 0; t < (n + TILE - 1) / TILE; t++) {
    uint32_t aCol = t * TILE + threadIdx.x;
    uint32_t bRow = t * TILE + threadIdx.y;
    // every thread participates: a syncthreads in a divergent if is a deadlock
    As[threadIdx.y][threadIdx.x] =
        (row < n && aCol < n) ? a[row * n + aCol] : 0.f;
    Bs[threadIdx.y][threadIdx.x] =
        (bRow < n && col < n) ? b[bRow * n + col] : 0.f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; k++)
      dot += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads(); // all readers done before the next tile overwrites As/Bs
  }

  if (row < n && col < n)
    c[row * n + col] = dot;
}

// Same algorithm, tile chosen at runtime. Array bound still cannot be `tile`,
// so the buffer is dynamic shared memory, sized by the 3rd <<<>>> argument.
// No unroll. Square block, so tile*tile <= 1024.
__global__ void mult_tiled_dyn(uint32_t n, uint32_t tile,
                               const float* a, const float* b, float* c) {
  extern __shared__ float smem[];
  float* As = smem;
  float* Bs = smem + tile * tile;

  uint32_t row = blockIdx.y * tile + threadIdx.y;
  uint32_t col = blockIdx.x * tile + threadIdx.x;
  float dot = 0;

  for (uint32_t t = 0; t < (n + tile - 1) / tile; t++) {
    uint32_t aCol = t * tile + threadIdx.x;
    uint32_t bRow = t * tile + threadIdx.y;
    As[threadIdx.y * tile + threadIdx.x] =
        (row < n && aCol < n) ? a[row * n + aCol] : 0.f;
    Bs[threadIdx.y * tile + threadIdx.x] =
        (bRow < n && col < n) ? b[bRow * n + col] : 0.f;
    __syncthreads();

    for (uint32_t k = 0; k < tile; k++)
      dot += As[threadIdx.y * tile + k] * Bs[k * tile + threadIdx.x];
    __syncthreads();
  }

  if (row < n && col < n)
    c[row * n + col] = dot;
}

int main() {
  const uint32_t n = 1024;
  constexpr int TILE = 32; // recompile to test 8, 16, 32
  matrix a(n,n), b(n,n), c(n,n);
  copy_to_device(a, b, c);

  dim3 block(TILE, TILE);
  dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
  mult_tiled<TILE><<<grid, block>>>(n, d_a, d_b, d_c);

  // runtime tile: bytes of shared memory are the 3rd launch argument
  // uint32_t tile = 16;
  // dim3 block(tile, tile);
  // dim3 grid((n + tile - 1) / tile, (n + tile - 1) / tile);
  // mult_tiled_dyn<<<grid, block, 2ull * tile * tile * sizeof(float)>>>(
  //     n, tile, d_a, d_b, d_c);
  copy_to_host
}
