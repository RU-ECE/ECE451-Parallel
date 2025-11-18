// matmul_kernels.cu
// Multiple kernels in one file for teaching/comparison
// Compile: nvcc -O3 matmul.cu

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
// #include <functional> // Removed std::function, so no need for this header

using namespace std;

#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        cudaError_t err = (call);                                                     \
        if (err != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(err));                                         \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

// Adjustable tile size (change to 16/32 to test)
#ifndef TILE
#define TILE 16
#endif

// Utility: set matrix (row-major) to constant 'k'
__global__ void set_matrix(float *A, int N, float k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    for (int i = idx; i < total; i += gridDim.x * blockDim.x)
        A[i] = k;
}
/*
suppose memory row size = 8k
N=1024
a[0] a[1] ...     a[1023]  (4096 bytes)
a[1024] = 4k
a[2048] = 8k (on a new page?)



*/
// ------------------------ TRANSPOSE KERNELS ------------------------

// 1 Naive transpose: each thread writes one element: ans[col*N + row] = a[row * N + col]
__global__ void transpose_naive(const float a[], float ans[], int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        ans[col * N + row] = a[row * N + col];
    }
}

// 2 Shared-memory tiled transpose with padding to avoid bank conflicts
__global__ void transpose_shared(const float a[], float ans[], int N) {
//    __shared__ float tile[TILE][TILE]; // this has bank conflicts
    __shared__ float tile[TILE][TILE + 1]; // +1 avoids bank conflicts
    int x = blockIdx.x * TILE + threadIdx.x; // Global column for input 'a'
    int y = blockIdx.y * TILE + threadIdx.y; // Global row for input 'a'

    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = a[y * N + x];
    //else
    //S    tile[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // write transposed
    // For the transposed output 'ans', the new row is the original column 'x'
    // and the new column is the original row 'y'.
    int out_row = blockIdx.x * TILE + threadIdx.x;
    int out_col = blockIdx.y * TILE + threadIdx.y;
    if (out_row < N && out_col < N)
        ans[out_row * N + out_col] = tile[threadIdx.y][threadIdx.x];
}

/*
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16


becomes
1 5 9 13
2 6 10 14
3 7 11 15
4 8 12 16

*/
// 2b Shared-memory transpose with coalesced writes
__global__ void transpose_shared_coalesced(const float a[], float ans[], int N) {
    __shared__ float tile[TILE][TILE + 1];

    int x_in = blockIdx.x * TILE + threadIdx.x;
    int y_in = blockIdx.y * TILE + threadIdx.y;

    if (x_in < N && y_in < N)
        tile[threadIdx.y][threadIdx.x] = a[y_in * N + x_in];

    __syncthreads();

    int x_out = blockIdx.y * TILE + threadIdx.x;
    int y_out = blockIdx.x * TILE + threadIdx.y;

    if (x_out < N && y_out < N)
        ans[y_out * N + x_out] = tile[threadIdx.x][threadIdx.y];
}

// 3 Register/unrolled tile transpose: each thread copies multiple elements using regs.
//    Good for moderate tile sizes; demonstrates using registers to hold temporaries.
//    This version treats block as TILE x TILE and each thread loops over a small stride in y.
__global__ void transpose_register_tile(const float a[], float ans[], int N) {
    const int bx = blockIdx.x * TILE;
    const int by = blockIdx.y * TILE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Each thread will copy a small column of the tile (unroll factor)
    const int UNROLL = 4; // tune: how many rows per thread (subject to TILE and blockDim)
    float regs[UNROLL];

    int baseY = by + ty * UNROLL;
    int x = bx + tx;

    // load into regs
    for (int u = 0; u < UNROLL; ++u) {
        int y = baseY + u;
        if (x < N && y < N)
            regs[u] = a[y * N + x];
        else
            regs[u] = 0.0f;
    }

    // write transposed: position becomes (x,y) -> (y,x)
    for (int u = 0; u < UNROLL; ++u) {
        int y = baseY + u;
        if (x < N && y < N)
            ans[x * N + y] = regs[u];
    }
}

// ------------------------ MATRIX MULTIPLICATION ------------------------

// 3a Naive matmul: one thread per output element C[row*N + col]
__global__ void matmul_naive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
        sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

// 3b) Tiled matmul using shared memory (classic)
__global__ void matmul_tiled(const float *A, const float *B, float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int nTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < nTiles; ++t) {
        int Arow = row;
        int Acol = t * TILE + threadIdx.x;
        int Brow = t * TILE + threadIdx.y;
        int Bcol = col;

        sA[threadIdx.y][threadIdx.x] = (Arow < N && Acol < N) ? A[Arow * N + Acol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (Brow < N && Bcol < N) ? B[Brow * N + Bcol] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = sum;
}

// 3c) Tiled with register tiling: each thread computes a small block (say 2x2) of C using registers
//      This demonstrates broadcasting/prefetch to registers for low-level optimization.
template<int UNROLL_COLS, int UNROLL_ROWS>
__global__ void matmul_tiled_register(const float *A, const float *B, float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int baseRow = blockIdx.y * TILE;
    int baseCol = blockIdx.x * TILE;

    int localRow = threadIdx.y * UNROLL_ROWS;
    int localCol = threadIdx.x * UNROLL_COLS;

    // Each thread will compute UNROLL_ROWS x UNROLL_COLS outputs
    float regs[UNROLL_ROWS][UNROLL_COLS];
    for (int i = 0; i < UNROLL_ROWS; ++i)
        for (int j = 0; j < UNROLL_COLS; ++j)
            regs[i][j] = 0.0f;

    int nTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < nTiles; ++t) {
        // Load tile blocks into shared memory.
        // Each thread (threadIdx.y, threadIdx.x) loads multiple elements
        // to fill the entire TILE x TILE shared memory block.
        for (int i_sh = threadIdx.y; i_sh < TILE; i_sh += blockDim.y) {
            int aRow_load = baseRow + i_sh; // Global row index for A
            int bRow_load = t * TILE + i_sh; // Global row index for B
            for (int j_sh = threadIdx.x; j_sh < TILE; j_sh += blockDim.x) {
                int aCol_load = t * TILE + j_sh; // Global col index for A
                int bCol_load = baseCol + j_sh; // Global col index for B

                sA[i_sh][j_sh] = (aRow_load < N && aCol_load < N) ? A[aRow_load * N + aCol_load] : 0.0f;
                sB[i_sh][j_sh] = (bRow_load < N && bCol_load < N) ? B[bRow_load * N + bCol_load] : 0.0f;
            }
        }
        __syncthreads();

        // compute using registers - inner k loop over TILE
        for (int k = 0; k < TILE; ++k) {
            float bvals[UNROLL_COLS];
            for (int jc = 0; jc < UNROLL_COLS; ++jc) {
                int colIdx = localCol + jc;
                bvals[jc] = sB[k][colIdx];
            }
            for (int ir = 0; ir < UNROLL_ROWS; ++ir) {
                float aval = sA[localRow + ir][k];
                for (int jc = 0; jc < UNROLL_COLS; ++jc)
                    regs[ir][jc] += aval * bvals[jc];
            }
        }
        __syncthreads();
    }

    // write results
    for (int ir = 0; ir < UNROLL_ROWS; ++ir) {
        int globalRow = baseRow + localRow + ir;
        for (int jc = 0; jc < UNROLL_COLS; ++jc) {
            int globalCol = baseCol + localCol + jc;
            if (globalRow < N && globalCol < N)
                C[globalRow * N + globalCol] = regs[ir][jc];
        }
    }
}

// 3d) Matmul assuming B is stored transposed (i.e., we pass Bt where Bt[col*N + k] = B[k*N + col])
//      This often improves locality because both accesses A[row*N + k] and Bt[col*N + k] are coalesced along k.
__global__ void matmul_Bt(const float *A, const float *Bt, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
        sum += A[row * N + k] * Bt[col * N + k]; // note Bt indexed by [col*N + k]
    C[row * N + col] = sum;
}

// ------------------------ HOST HELPERS ------------------------

__host__ void fill_host(float *h, int N, float v) {
    for (int i = 0; i < N * N; ++i) h[i] = v;
}

__host__ bool approx_equal(const float *a, const float *b, int N, float tol = 1e-3f) {
    for (int i = 0; i < N * N; ++i) {
        float A = a[i], B = b[i];
        if (fabs(A - B) > tol * fmax(1.0f, fabs(A)))
            return false;
    }
    return true;
}

// transpose on host for correctness check
__host__ void transpose_host(const float *A, float *At, int N) {
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            At[c * N + r] = A[r * N + c];
}

__host__ void report_test(const char *name, bool ok) {
    if (!ok) printf("%s FAILED\n", name);
}

__host__ void print_perf_line(const char *name, float ms, double metric, const char *unit) {
    printf("%-32s %6.3f ms %8.3f %s\n", name, ms, metric, unit);
}

template<typename F>
float benchmark_kernel(F launch_and_sync, int iterations = 10) {
    // warmup + timed runs using CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    launch_and_sync(); // warmup
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch_and_sync();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

template<typename F>
float benchmark_and_report(const char *name,
                           F launch_and_sync,
                           double scale_value,
                           const char *unit,
                           int iterations = 10) {
    float ms = benchmark_kernel(launch_and_sync, iterations);
    double metric = (ms > 0.0f) ? scale_value / (ms * 1e-3) : 0.0;
    print_perf_line(name, ms, metric, unit);
    return ms;
}

int main(int argc, char **argv) {
    int N = (argc >= 2) ? atoi(argv[1]) : 1024;
    printf("N = %d, tile = %d\n", N, TILE);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    float *hCref = (float*)malloc(bytes);
    float *hBt = (float*)malloc(bytes);

    // init host
    // fill_host(hA, N, 1.0f); // A = 1.0
    for (int i = 0; i < N * N; ++i) hA[i] = (float)i;
    fill_host(hB, N, 2.0f); // B = 2.0

    float *dA, *dB, *dC, *d_tmp;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));
    CHECK_CUDA(cudaMalloc(&d_tmp, bytes)); // scratch (transpose result etc)

    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice)); // Corrected: HostToDevice

    // Prepare grid/block sizes
    const int REG_UNROLL = 4;
    constexpr int MATMUL_UC = 2;
    constexpr int MATMUL_UR = 2;
    if (TILE % REG_UNROLL != 0) {
        fprintf(stderr, "TILE=%d not divisible by %d for transpose_register_tile\n", TILE, REG_UNROLL);
        return 1;
    }
    if (TILE % MATMUL_UR != 0 || TILE % MATMUL_UC != 0) {
        fprintf(stderr, "TILE=%d incompatible with MATMUL_UR=%d MATMUL_UC=%d\n", TILE, MATMUL_UR, MATMUL_UC);
        return 1;
    }

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    dim3 regBlock(TILE, TILE / REG_UNROLL);

    // --- Test transpose kernels ---
    // Naive transpose
    transpose_naive<<<grid, block>>>(dA, d_tmp, N);
    CHECK_CUDA(cudaMemcpy(hC, d_tmp, bytes, cudaMemcpyDeviceToHost));
    // compute host transpose and compare
    transpose_host(hA, hBt, N);
    bool ok = approx_equal(hC, hBt, N);
    report_test("transpose_naive", ok);

    // Shared-memory transpose
    transpose_shared<<<grid, block>>>(dA, d_tmp, N);
    CHECK_CUDA(cudaMemcpy(hC, d_tmp, bytes, cudaMemcpyDeviceToHost));
    ok = approx_equal(hC, hBt, N);
    report_test("transpose_shared", ok);

    // Shared-memory transpose with coalesced writes
    transpose_shared_coalesced<<<grid, block>>>(dA, d_tmp, N);
    CHECK_CUDA(cudaMemcpy(hC, d_tmp, bytes, cudaMemcpyDeviceToHost));
    ok = approx_equal(hC, hBt, N);
    report_test("transpose_shared_coalesced", ok);

    // Register/unroll transpose - adjust launch so each thread handles UNROLL rows
    // For the register kernel we used UNROLL=4 and thread block dims should be TILE/UNROLL x TILE
    {
        transpose_register_tile<<<grid, regBlock>>>(dA, d_tmp, N);
        CHECK_CUDA(cudaMemcpy(hC, d_tmp, bytes, cudaMemcpyDeviceToHost));
        ok = approx_equal(hC, hBt, N);
        report_test("transpose_register_tile", ok);
    }

    const double bytes_per_transpose = 2.0 * static_cast<double>(bytes); // reads and writes once per element
    const double gb_per_transpose = bytes_per_transpose / (1024.0 * 1024.0 * 1024.0);

    transpose_host(hA, hBt, N);

    benchmark_and_report("transpose_naive", [&]() {
        transpose_naive<<<grid, block>>>(dA, d_tmp, N);
        CHECK_CUDA(cudaPeekAtLastError());
    }, gb_per_transpose, "GB/s", 5);

    benchmark_and_report("transpose_shared", [&]() {
        transpose_shared<<<grid, block>>>(dA, d_tmp, N);
        CHECK_CUDA(cudaPeekAtLastError());
    }, gb_per_transpose, "GB/s", 5);

    benchmark_and_report("transpose_shared_coalesced", [&]() {
        transpose_shared_coalesced<<<grid, block>>>(dA, d_tmp, N);
        CHECK_CUDA(cudaPeekAtLastError());
    }, gb_per_transpose, "GB/s", 5);

    benchmark_and_report("transpose_register_tile", [&]() {
        transpose_register_tile<<<grid, regBlock>>>(dA, d_tmp, N);
        CHECK_CUDA(cudaPeekAtLastError());
    }, gb_per_transpose, "GB/s", 5);

    // --- Test matmul kernels ---
    // Reference: simple CPU matmul into hCref
    // For speed, since matrices are constant we can compute analytically: A all ones, B all twos => C entries = 1*2*N = 2*N
    // but keep general host multiply for clarity (only for small N — be careful)
    for (int i = 0; i < N * N; ++i) hCref[i] = 0.0f;
    for (int r = 0; r < N; ++r)
        for (int k = 0; k < N; ++k)
            for (int c = 0; c < N; ++c)
                hCref[r * N + c] += hA[r * N + k] * hB[k * N + c];

    // 3a naive matmul
    matmul_naive<<<grid, block>>>(dA, dB, dC, N);
    CHECK_CUDA(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    report_test("matmul_naive", approx_equal(hC, hCref, N));

    // 3b tiled matmul
    matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
    CHECK_CUDA(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    report_test("matmul_tiled", approx_equal(hC, hCref, N));

    // 3c tiled_register: choose UNROLL dims that fit TILE
    // e.g., UNROLL_COLS=2, UNROLL_ROWS=2 and we need blockDim = TILE/UNROLL_ROWS, TILE/UNROLL_COLS
    dim3 matmulGrid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    dim3 matmulBlock(TILE, TILE);
    dim3 matmulRegBlock(TILE/MATMUL_UR, TILE/MATMUL_UC);

    {
        matmul_tiled_register<MATMUL_UC,MATMUL_UR><<<matmulGrid, matmulRegBlock>>>(dA, dB, dC, N);
        CHECK_CUDA(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
        report_test("matmul_tiled_register", approx_equal(hC, hCref, N));
    }

    // 3d matmul with B transposed: create Bt on device and run
    transpose_shared<<<grid, block>>>(dB, d_tmp, N);
    // now d_tmp holds B^T
    matmul_Bt<<<matmulGrid, matmulBlock>>>(dA, d_tmp, dC, N);
    CHECK_CUDA(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    report_test("matmul_Bt", approx_equal(hC, hCref, N));

    // Example micro-benchmark: time matmul_tiled
    double matmul_work_giga = 2.0 * (double)N * N * N / 1e9;

    benchmark_and_report("matmul_naive", [&]() {
        matmul_naive<<<matmulGrid, matmulBlock>>>(dA, dB, dC, N);
        CHECK_CUDA(cudaPeekAtLastError());
    }, matmul_work_giga, "GFLOPS", 5);

    benchmark_and_report("matmul_tiled", [&]() {
        matmul_tiled<<<matmulGrid, matmulBlock>>>(dA, dB, dC, N);
        CHECK_CUDA(cudaPeekAtLastError());
    }, matmul_work_giga, "GFLOPS", 5);

    benchmark_and_report("matmul_tiled_register", [&]() {
        matmul_tiled_register<MATMUL_UC,MATMUL_UR><<<matmulGrid, matmulRegBlock>>>(dA, dB, dC, N);
        CHECK_CUDA(cudaPeekAtLastError());
    }, matmul_work_giga, "GFLOPS", 5);

    // cleanup
    free(hA); free(hB); free(hC); free(hCref); free(hBt);
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC)); CHECK_CUDA(cudaFree(d_tmp));

    return 0;
}