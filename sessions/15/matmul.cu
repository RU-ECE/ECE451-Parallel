// matmul_kernels.cu
// Multiple kernels in one file for teaching/comparison
// Compile: nvcc -O3 matmul.cu

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

#define CHECK_CUDA(call)                                                                                               \
	do {                                                                                                               \
		cudaError_t err = (call);                                                                                      \
		if (err != cudaSuccess) {                                                                                      \
			fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                    \
			exit(1);                                                                                                   \
		}                                                                                                              \
	} while (0)

// Adjustable tile size (change to 16/32 to test)
#ifndef TILE
#define TILE 16
#endif

// Utility: set matrix (row-major) to constant 'k'
__global__ void set_matrix(float* A, const long N, const float k) {
	const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	const auto total = N * N;
	for (auto i = idx; i < total; i += gridDim.x * blockDim.x)
		A[i] = k;
}
/*
 * suppose memory row size = 8k
 * N=1024
 * a[0], a[1], ..., a[1023] (4096 bytes)
 * a[1024] = 4k
 * a[2048] = 8k (on a new page?)
 */
// ------------------------ TRANSPOSE KERNELS ------------------------

// 1	Naive transpose: each thread writes one element: ans[col*N + row] = a[row * N + col]
__global__ void transpose_naive(const float a[], float ans[], const long N) {
	if (const auto row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
		row < N && col < N) {
		ans[col * N + row] = a[row * N + col];
	}
}

// 2	Shared-memory tiled transpose with padding to avoid bank conflicts
__global__ void transpose_shared(const float a[], float ans[], const long N) {
	// __shared__ float tile[TILE][TILE]; // this has bank conflicts
	__shared__ float tile[TILE][TILE + 1]; // +1 avoids bank conflicts
	if (const auto x = blockIdx.x * TILE + threadIdx.x, // Global column for input 'a'
		y = blockIdx.y * TILE + threadIdx.y; // Global row for input 'a'
		x < N && y < N) {
		tile[threadIdx.y][threadIdx.x] = a[y * N + x];
	} /*else {
		tile[threadIdx.y][threadIdx.x] = 0.0f;
	}*/
	__syncthreads();

	// write transposed
	// For the transposed output 'ans', the new row is the original column 'x'
	// and the new column is the original row 'y'.
	if (const auto out_row = blockIdx.x * TILE + threadIdx.x, out_col = blockIdx.y * TILE + threadIdx.y;
		out_row < N && out_col < N) {
		ans[out_row * N + out_col] = tile[threadIdx.y][threadIdx.x];
	}
}

/*
 * 1 2 3 4
 * 5 6 7 8
 * 9 10 11 12
 * 13 14 15 16
 *
 *
 * becomes
 * 1 5 9 13
 * 2 6 10 14
 * 3 7 11 15
 * 4 8 12 16
 */
// 2b	Shared-memory transpose with coalesced writes
__global__ void transpose_shared_coalesced(const float a[], float ans[], const long N) {
	__shared__ float tile[TILE][TILE + 1];
	if (const auto x_in = blockIdx.x * TILE + threadIdx.x, y_in = blockIdx.y * TILE + threadIdx.y; x_in < N && y_in < N)
		tile[threadIdx.y][threadIdx.x] = a[y_in * N + x_in];
	__syncthreads();
	if (const auto x_out = blockIdx.y * TILE + threadIdx.x, y_out = blockIdx.x * TILE + threadIdx.y;
		x_out < N && y_out < N) {
		ans[y_out * N + x_out] = tile[threadIdx.x][threadIdx.y];
	}
}

// 3	Register/unrolled tile transpose: each thread copies multiple elements using regs.
//		Good for moderate tile sizes; demonstrates using registers to hold temporaries.
//		This version treats block as TILE x TILE and each thread loops over a small stride in y.
__global__ void transpose_register_tile(const float a[], float ans[], const long N) {
	const auto bx = blockIdx.x * TILE, by = blockIdx.y * TILE, tx = threadIdx.x, ty = threadIdx.y;
	// Each thread will copy a small column of the tile (unroll factor)
	constexpr auto UNROLL = 4; // tune: how many rows per thread (subject to TILE and blockDim)
	float regs[UNROLL];
	const auto baseY = by + ty * UNROLL, x = bx + tx;
	// load into regs
	for (auto u = 0; u < UNROLL; ++u)
		regs[u] = x < N && baseY + u < N ? a[(baseY + u) * N + x] : 0.0f;
	// write transposed: position becomes (x,y) -> (y,x)
	for (auto u = 0; u < UNROLL; ++u)
		if (x < N && baseY + u < N)
			ans[x * N + baseY + u] = regs[u];
}

// ------------------------ MATRIX MULTIPLICATION ------------------------

// 3a	Naive matmul: one thread per output element C[row*N + col]
__global__ void matmul_naive(const float* A, const float* B, float* C, const long N) {
	const auto row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= N || col >= N)
		return;
	auto sum = 0.0f;
	for (auto k = 0; k < N; ++k)
		sum += A[row * N + k] * B[k * N + col];
	C[row * N + col] = sum;
}

// 3b	Tiled matmul using shared memory (classic)
__global__ void matmul_tiled(const float* A, const float* B, float* C, const long N) {
	__shared__ float sA[TILE][TILE];
	__shared__ float sB[TILE][TILE];
	const auto row = blockIdx.y * TILE + threadIdx.y, col = blockIdx.x * TILE + threadIdx.x;
	auto sum = 0.0f;
	const auto nTiles = (N + TILE - 1) / TILE;
	for (auto t = 0; t < nTiles; ++t) {
		const auto a_row = row, a_col = t * TILE + threadIdx.x, b_row = t * TILE + threadIdx.y, b_col = col;
		sA[threadIdx.y][threadIdx.x] = a_row < N && a_col < N ? A[a_row * N + a_col] : 0.0f;
		sB[threadIdx.y][threadIdx.x] = b_row < N && b_col < N ? B[b_row * N + b_col] : 0.0f;
		__syncthreads();
#pragma unroll
		for (auto k = 0; k < TILE; ++k)
			sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
		__syncthreads();
	}
	if (row < N && col < N)
		C[row * N + col] = sum;
}

// 3c	Tiled with register tiling: each thread computes a small block (say 2x2) of C using registers
//		This demonstrates broadcasting/prefetch to registers for low-level optimization.
template <int UNROLL_COLS, int UNROLL_ROWS>
__global__ void matmul_tiled_register(const float* A, const float* B, float* C, const long N) {
	__shared__ float sA[TILE][TILE];
	__shared__ float sB[TILE][TILE];
	const auto baseRow = blockIdx.y * TILE, baseCol = blockIdx.x * TILE, localRow = threadIdx.y * UNROLL_ROWS,
			   localCol = threadIdx.x * UNROLL_COLS;
	// Each thread will compute UNROLL_ROWS x UNROLL_COLS outputs
	float regs[UNROLL_ROWS][UNROLL_COLS];
	for (auto i = 0; i < UNROLL_ROWS; ++i)
		for (auto j = 0; j < UNROLL_COLS; ++j)
			regs[i][j] = 0.0f;
	const auto nTiles = (N + TILE - 1) / TILE;
	for (auto t = 0; t < nTiles; ++t) {
		// Load tile blocks into shared memory.
		// Each thread (threadIdx.y, threadIdx.x) loads multiple elements
		// to fill the entire TILE x TILE shared memory block.
		for (auto i_sh = threadIdx.y; i_sh < TILE; i_sh += blockDim.y) {
			for (auto j_sh = threadIdx.x,
					  aRow_load = baseRow + i_sh, // Global row index for A
				 bRow_load = t * TILE + i_sh; // Global row index for B
				 j_sh < TILE; j_sh += blockDim.x) {
				const auto aCol_load = t * TILE + j_sh, // Global col index for A
					bCol_load = baseCol + j_sh; // Global col index for B
				sA[i_sh][j_sh] = aRow_load < N && aCol_load < N ? A[aRow_load * N + aCol_load] : 0.0f;
				sB[i_sh][j_sh] = bRow_load < N && bCol_load < N ? B[bRow_load * N + bCol_load] : 0.0f;
			}
		}
		__syncthreads();
		// compute using registers - inner k loop over TILE
		for (auto k = 0; k < TILE; ++k) {
			float bvals[UNROLL_COLS];
			for (auto jc = 0; jc < UNROLL_COLS; ++jc)
				bvals[jc] = sB[k][localCol + jc];
			for (auto ir = 0; ir < UNROLL_ROWS; ++ir) {
				auto aval = sA[localRow + ir][k];
				for (auto jc = 0; jc < UNROLL_COLS; ++jc)
					regs[ir][jc] += aval * bvals[jc];
			}
		}
		__syncthreads();
	}
	// write results
	for (auto ir = 0; ir < UNROLL_ROWS; ++ir) {
		const auto globalRow = baseRow + localRow + ir;
		for (auto jc = 0; jc < UNROLL_COLS; ++jc)
			if (const auto globalCol = baseCol + localCol + jc; globalRow < N && globalCol < N)
				C[globalRow * N + globalCol] = regs[ir][jc];
	}
}

// 3d	Matmul assuming B is stored transposed (i.e., we pass Bt where Bt[col*N + k] = B[k*N + col])
//		This often improves locality because both accesses A[row*N + k] and Bt[col*N + k] are coalesced along k.
__global__ void matmul_Bt(const float* A, const float* Bt, float* C, const long N) {
	const auto row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= N || col >= N)
		return;
	auto sum = 0.0f;
	for (auto k = 0; k < N; ++k)
		sum += A[row * N + k] * Bt[col * N + k]; // note Bt indexed by [col*N + k]
	C[row * N + col] = sum;
}

// ------------------------ HOST HELPERS ------------------------

__host__ void fill_host(float* h, const long N, const float v) {
	for (auto i = 0; i < N * N; ++i)
		h[i] = v;
}

__host__ bool approx_equal(const float* a, const float* b, const long N, const float tol = 1e-3f) {
	for (auto i = 0; i < N * N; ++i)
		if (fabs(a[i] - b[i]) > tol * fmax(1.0f, fabs(a[i])))
			return false;
	return true;
}

// transpose on host for correctness check
__host__ void transpose_host(const float* A, float* At, const long N) {
	for (auto r = 0; r < N; ++r)
		for (auto c = 0; c < N; ++c)
			At[c * N + r] = A[r * N + c];
}

__host__ void report_test(const char* name, const bool ok) {
	if (!ok)
		printf("%s FAILED\n", name);
}

__host__ void print_perf_line(const char* name, const float ms, const double metric, const char* unit) {
	printf("%-32s %6.3f ms %8.3f %s\n", name, ms, metric, unit);
}

template <typename F>
float benchmark_kernel(F launch_and_sync, const int iterations = 10) {
	// warmup + timed runs using CUDA events
	cudaEvent_t start, stop;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));
	launch_and_sync(); // warmup
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaEventRecord(start));
	for (auto i = 0; i < iterations; ++i)
		launch_and_sync();
	CHECK_CUDA(cudaEventRecord(stop));
	CHECK_CUDA(cudaEventSynchronize(stop));
	auto ms = 0.0f;
	CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
	ms /= static_cast<float>(iterations);
	CHECK_CUDA(cudaEventDestroy(start));
	CHECK_CUDA(cudaEventDestroy(stop));
	return ms;
}

template <typename F>
float benchmark_and_report(const char* name, F launch_and_sync, const double scale_value, const char* unit,
						   int iterations = 10) {
	const auto ms = benchmark_kernel(launch_and_sync, iterations);
	const auto metric = ms > 0.0f ? scale_value / (ms * 1e-3) : 0.0;
	print_perf_line(name, ms, metric, unit);
	return ms;
}

int main(const int argc, char* argv[]) {
	const auto N = argc >= 2 ? strtol(argv[1], nullptr, 10) : 1024;
	printf("N = %ld, tile = %d\n", N, TILE);
	const auto bytes = N * N * sizeof(float);
	const auto hA = static_cast<float*>(malloc(bytes)), hB = static_cast<float*>(malloc(bytes)),
			   hC = static_cast<float*>(malloc(bytes)), hCref = static_cast<float*>(malloc(bytes)),
			   hBt = static_cast<float*>(malloc(bytes));
	// init host
	// fill_host(hA, N, 1.0f); // A = 1.0
	for (auto i = 0; i < N * N; ++i)
		hA[i] = static_cast<float>(i);
	fill_host(hB, N, 2.0f); // B = 2.0
	float *dA, *dB, *dC, *d_tmp;
	CHECK_CUDA(cudaMalloc(&dA, bytes));
	CHECK_CUDA(cudaMalloc(&dB, bytes));
	CHECK_CUDA(cudaMalloc(&dC, bytes));
	CHECK_CUDA(cudaMalloc(&d_tmp, bytes)); // scratch (transpose result etc)
	CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice)); // Corrected: HostToDevice
	// Prepare grid/block sizes
	constexpr auto REG_UNROLL = 4, MATMUL_UC = 2, MATMUL_UR = 2;
	if constexpr (TILE % REG_UNROLL != 0) {
		fprintf(stderr, "TILE=%d not divisible by %d for transpose_register_tile\n", TILE, REG_UNROLL);
		return 1;
	}
	if constexpr (TILE % MATMUL_UC != 0) {
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
	const auto bytes_per_transpose = 2 * static_cast<double>(bytes), // reads and writes once per element
		gb_per_transpose = bytes_per_transpose / (1024.0 * 1024.0 * 1024.0);
	transpose_host(hA, hBt, N);
	benchmark_and_report(
		"transpose_naive",
		[&] {
			transpose_naive<<<grid, block>>>(dA, d_tmp, N);
			CHECK_CUDA(cudaPeekAtLastError());
		},
		gb_per_transpose, "GB/s", 5);
	benchmark_and_report(
		"transpose_shared",
		[&] {
			transpose_shared<<<grid, block>>>(dA, d_tmp, N);
			CHECK_CUDA(cudaPeekAtLastError());
		},
		gb_per_transpose, "GB/s", 5);
	benchmark_and_report(
		"transpose_shared_coalesced",
		[&] {
			transpose_shared_coalesced<<<grid, block>>>(dA, d_tmp, N);
			CHECK_CUDA(cudaPeekAtLastError());
		},
		gb_per_transpose, "GB/s", 5);
	benchmark_and_report(
		"transpose_register_tile",
		[&] {
			transpose_register_tile<<<grid, regBlock>>>(dA, d_tmp, N);
			CHECK_CUDA(cudaPeekAtLastError());
		},
		gb_per_transpose, "GB/s", 5);
	// --- Test matmul kernels ---
	// Reference: simple CPU matmul into hCref
	// For speed, since matrices are constant we can compute analytically: A all ones, B all twos => C entries = 1*2*N =
	// 2*N but keep general host multiply for clarity (only for small N â€” be careful)
	for (auto i = 0; i < N * N; ++i)
		hCref[i] = 0.0f;
	for (auto r = 0; r < N; ++r) {
		for (auto k = 0; k < N; ++k)
			for (auto c = 0; c < N; ++c)
				hCref[r * N + c] += hA[r * N + k] * hB[k * N + c];
	}
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
	dim3 matmulRegBlock(TILE / MATMUL_UR, TILE / MATMUL_UC);
	{
		matmul_tiled_register<MATMUL_UC, MATMUL_UR><<<matmulGrid, matmulRegBlock>>>(dA, dB, dC, N);
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
	const auto matmul_work_giga = 2 * N * N * N / 1e9;
	benchmark_and_report(
		"matmul_naive",
		[&] {
			matmul_naive<<<matmulGrid, matmulBlock>>>(dA, dB, dC, N);
			CHECK_CUDA(cudaPeekAtLastError());
		},
		matmul_work_giga, "GFLOPS", 5);
	benchmark_and_report(
		"matmul_tiled",
		[&] {
			matmul_tiled<<<matmulGrid, matmulBlock>>>(dA, dB, dC, N);
			CHECK_CUDA(cudaPeekAtLastError());
		},
		matmul_work_giga, "GFLOPS", 5);
	benchmark_and_report(
		"matmul_tiled_register",
		[&] {
			matmul_tiled_register<MATMUL_UC, MATMUL_UR><<<matmulGrid, matmulRegBlock>>>(dA, dB, dC, N);
			CHECK_CUDA(cudaPeekAtLastError());
		},
		matmul_work_giga, "GFLOPS", 5);
	// cleanup
	free(hA);
	free(hB);
	free(hC);
	free(hCref);
	free(hBt);
	CHECK_CUDA(cudaFree(dA));
	CHECK_CUDA(cudaFree(dB));
	CHECK_CUDA(cudaFree(dC));
	CHECK_CUDA(cudaFree(d_tmp));
	return 0;
}
