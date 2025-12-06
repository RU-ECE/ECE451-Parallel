// blur3x3_bench.cu
// Compile: nvcc -O3 blur3x3_bench.cu -o blur3x3_bench
// Run: ./blur3x3_bench [width] [height] [iters]

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

using namespace std;

#define CHECK(call)                                                                                                    \
	do {                                                                                                               \
		cudaError_t err = (call);                                                                                      \
		if (err != cudaSuccess) {                                                                                      \
			fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                    \
			exit(1);                                                                                                   \
		}                                                                                                              \
	} while (0)

static constexpr float KERNEL_3x3[3][3] = {
	{1 / 9.0f, 1 / 9.0f, 1 / 9.0f}, {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}, {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}};

/*
 * P P P
 * P C P
 * P P P
 * C is the center pixel
 *
 * 1 2 3 4				1+2+3+2+3+5+3+5+8 = 32 / 9 = 3.55
 * 2 3 5 6				2+3+5+3+5+8+5+8+9 = 48 / 9 = 5.333333333333333
 * 3 5 8 9
 * 4 6 5 1
 *
 *
 * 3.55
 */

// -----------------------------
// Kernel 1: Naive brute-force
// Each thread computes one output pixel reading 9 global loads
// -----------------------------
__global__ void blur3x3_naive(const float* __restrict__ in, float* __restrict__ out, const int W, const int H) {
	const auto x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H)
		return;
	auto sum = 0.0f;
	// 3x3 kernel centered on (x,y). Use clamp at borders.
	for (auto ky = -1; ky <= 1; ++ky)
		for (auto kx = -1; kx <= 1; ++kx)
			sum += in[min(max(y + ky, 0), H - 1) * W + min(max(x + kx, 0), W - 1)] * KERNEL_3x3[ky + 1][kx + 1];
	out[y * W + x] = sum;
}

// -----------------------------
// Kernel 2: Shared-memory tiling (one output per thread)
// - Each block loads tile of size (BX x BY) plus 1-pixel halo around
// - Each thread loads its own interior pixel and boundary threads load halo
// -----------------------------
template <int BX, int BY>
__global__ void blur3x3_smem(const float* __restrict__ in, float* __restrict__ out, const int W, const int H) {
	// shared tile dimensions: BX+2 by BY+2
	__shared__ float s[(BY + 2) * (BX + 2)];
	const auto tx = threadIdx.x, ty = threadIdx.y, x = blockIdx.x * BX + tx, y = blockIdx.y * BY + ty;
	// shared memory coordinates (shifted by +1 for halo)
	const auto s_x = tx + 1, s_y = ty + 1;
	auto s_idx = s_y * (BX + 2) + s_x;
	// load center element (if within image) else load clamped border
	const auto xc = min(max(x, 0), W - 1), yc = min(max(y, 0), H - 1);
	s[s_idx] = in[yc * W + xc];
	// load halos: threads on edges load the surrounding pixels
	// left
	if (tx == 0)
		s[s_y * (BX + 2) + (s_x - 1)] = in[yc * W + min(max(x - 1, 0), W - 1)];
	// right
	if (tx == BX - 1)
		s[s_y * (BX + 2) + (s_x + 1)] = in[yc * W + min(max(x + 1, 0), W - 1)];
	// top
	if (ty == 0)
		s[(s_y - 1) * (BX + 2) + s_x] = in[min(max(y - 1, 0), H - 1) * W + xc];
	// bottom
	if (ty == BY - 1)
		s[(s_y + 1) * (BX + 2) + s_x] = in[min(max(y + 1, 0), H - 1) * W + xc];
	// corners (four)
	if (tx == 0 && ty == 0)
		s[(s_y - 1) * (BX + 2) + (s_x - 1)] = in[min(max(y - 1, 0), H - 1) * W + min(max(x - 1, 0), W - 1)];
	if (tx == BX - 1 && ty == 0)
		s[(s_y - 1) * (BX + 2) + (s_x + 1)] = in[min(max(y - 1, 0), H - 1) * W + min(max(x + 1, 0), W - 1)];
	if (tx == 0 && ty == BY - 1)
		s[(s_y + 1) * (BX + 2) + (s_x - 1)] = in[min(max(y + 1, 0), H - 1) * W + min(max(x - 1, 0), W - 1)];
	if (tx == BX - 1 && ty == BY - 1)
		s[(s_y + 1) * (BX + 2) + (s_x + 1)] = in[min(max(y + 1, 0), H - 1) * W + min(max(x + 1, 0), W - 1)];
	__syncthreads();
	// compute output if within image bounds
	if (x < W && y < H) {
		auto sum = 0.0f;
		// unrolled 3x3 loop reading from shared memory
		sum += s[(s_y - 1) * (BX + 2) + (s_x - 1)] * KERNEL_3x3[0][0];
		sum += s[(s_y - 1) * (BX + 2) + s_x] * KERNEL_3x3[0][1];
		sum += s[(s_y - 1) * (BX + 2) + (s_x + 1)] * KERNEL_3x3[0][2];
		sum += s[s_y * (BX + 2) + (s_x - 1)] * KERNEL_3x3[1][0];
		sum += s[s_y * (BX + 2) + s_x] * KERNEL_3x3[1][1];
		sum += s[s_y * (BX + 2) + (s_x + 1)] * KERNEL_3x3[1][2];
		sum += s[(s_y + 1) * (BX + 2) + (s_x - 1)] * KERNEL_3x3[2][0];
		sum += s[(s_y + 1) * (BX + 2) + s_x] * KERNEL_3x3[2][1];
		sum += s[(s_y + 1) * (BX + 2) + (s_x + 1)] * KERNEL_3x3[2][2];
		out[y * W + x] = sum;
	}
}

// -----------------------------
// Kernel 3: Shared-memory + register blocking
// Each thread computes TWO horizontal outputs (when possible).
// Grid.x is sized so each block handles a tile width = BX*2.
// Shared memory width = BX*2 + 2
// -----------------------------
template <int BX, int BY>
__global__ void blur3x3_smem_reg(const float* __restrict__ in, float* __restrict__ out, const int W, const int H) {
	// tile width handled by block = BX*2, tile height = BY
	const auto tileW = BX * 2;
	const auto tileH = BY;
	const auto sW = tileW + 2;
	[[maybe_unused]] const auto sH = tileH + 2;
	extern __shared__ float s_ext[]; // size sW * sH
	const auto s = s_ext;
	const auto tx = threadIdx.x, // 0..BX-1
		ty = threadIdx.y; // 0..BY-1
	// base coordinates for block
	const auto baseX = blockIdx.x * tileW, baseY = blockIdx.y * tileH;
	// each thread will load two interior pixels (x0,x1) into shared memory
	const auto x0 = baseX + tx,
			   x1 = baseX + tx + BX, // second column handled by same thread
		y = baseY + ty;
	// shared memory coords: shift by +1 (halo)
	const auto s_x0 = tx + 1, s_x1 = tx + 1 + BX, s_y = ty + 1;
	// load center pixels (clamped)
	const auto yc = min(max(y, 0), H - 1);
	// load center two pixels (if they exist)
	s[s_y * sW + s_x0] = x0 < W ? in[yc * W + min(max(x0, 0), W - 1)] : 0.0f;
	s[s_y * sW + s_x1] = x1 < W ? in[yc * W + min(max(x1, 0), W - 1)] : 0.0f;
	// load left/right halo columns: threads with tx==0 load left halo column (two spots)
	if (tx == 0) {
		// left halo (at s_x0 -1)
		s[s_y * sW + 0] = in[yc * W + min(max(baseX - 1, 0), W - 1)];
		// rightmost interior location was loaded by thread tx=BX-1's x1 slot; but we should ensure right halo loaded at
		// s_x = tileW+1
		s[s_y * sW + (sW - 1)] = in[yc * W + min(max(baseX + tileW - 1, 0), W - 1)];
	}
	// load top/bottom halo rows (for this thread's two x positions)
	if (ty == 0) {
		const auto yt = min(max(y - 1, 0), H - 1);
		// top for x0
		s[(s_y - 1) * sW + s_x0] = x0 < W ? in[yt * W + min(max(x0, 0), W - 1)] : 0.0f;
		// top for x1
		s[(s_y - 1) * sW + s_x1] = x1 < W ? in[yt * W + min(max(x1, 0), W - 1)] : 0.0f;
		// corners: left-top and right-top (for tx==0 thread)
		if (tx == 0) {
			s[(s_y - 1) * sW + 0] = in[yt * W + min(max(baseX - 1, 0), W - 1)];
			s[(s_y - 1) * sW + (sW - 1)] = in[yt * W + min(max(baseX + tileW - 1, 0), W - 1)];
		}
	}
	if (ty == BY - 1) {
		const auto yb = min(max(y + 1, 0), H - 1);
		s[(s_y + 1) * sW + s_x0] = x0 < W ? in[yb * W + min(max(x0, 0), W - 1)] : 0.0f;
		s[(s_y + 1) * sW + s_x1] = x1 < W ? in[yb * W + min(max(x1, 0), W - 1)] : 0.0f;
		if (tx == 0) {
			s[(s_y + 1) * sW + 0] = in[yb * W + min(max(baseX - 1, 0), W - 1)];
			s[(s_y + 1) * sW + (sW - 1)] = in[yb * W + min(max(baseX + tileW - 1, 0), W - 1)];
		}
	}
	__syncthreads();
	// Now each thread computes two outputs (for x0 and x1) using registers
	if (x0 < W && y < H) {
		out[y * W + x0] = s[(s_y - 1) * sW + (s_x0 - 1)] * KERNEL_3x3[0][0] +
			s[(s_y - 1) * sW + s_x0] * KERNEL_3x3[0][1] + s[(s_y - 1) * sW + (s_x0 + 1)] * KERNEL_3x3[0][2] +
			s[s_y * sW + (s_x0 - 1)] * KERNEL_3x3[1][0] + s[s_y * sW + s_x0] * KERNEL_3x3[1][1] +
			s[s_y * sW + (s_x0 + 1)] * KERNEL_3x3[1][2] + s[(s_y + 1) * sW + (s_x0 - 1)] * KERNEL_3x3[2][0] +
			s[(s_y + 1) * sW + s_x0] * KERNEL_3x3[2][1] + s[(s_y + 1) * sW + (s_x0 + 1)] * KERNEL_3x3[2][2];
	}
	// second output
	if (x1 < W && y < H) {
		out[y * W + x1] = s[(s_y - 1) * sW + (s_x1 - 1)] * KERNEL_3x3[0][0] +
			s[(s_y - 1) * sW + s_x1] * KERNEL_3x3[0][1] + s[(s_y - 1) * sW + (s_x1 + 1)] * KERNEL_3x3[0][2] +
			s[s_y * sW + (s_x1 - 1)] * KERNEL_3x3[1][0] + s[s_y * sW + s_x1] * KERNEL_3x3[1][1] +
			s[s_y * sW + (s_x1 + 1)] * KERNEL_3x3[1][2] + s[(s_y + 1) * sW + (s_x1 - 1)] * KERNEL_3x3[2][0] +
			s[(s_y + 1) * sW + s_x1] * KERNEL_3x3[2][1] + s[(s_y + 1) * sW + (s_x1 + 1)] * KERNEL_3x3[2][2];
	}
}

// -----------------------------
// Host helpers: init, verify, benchmark
// -----------------------------
void init_image(float* a, const int W, const int H) {
	for (auto y = 0; y < H; ++y)
		for (auto x = 0; x < W; ++x)
			a[y * W + x] = static_cast<float>((x * 37 ^ y * 17) & 255) / 255.0f; // something non-trivial
}

double benchmark_naive(const float* d_in, float* d_out, const int W, const int H, const int iters) {
	dim3 block(16, 16);
	dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
	cudaEvent_t s, e;
	CHECK(cudaEventCreate(&s));
	CHECK(cudaEventCreate(&e));
	CHECK(cudaEventRecord(s));
	for (auto i = 0; i < iters; ++i)
		blur3x3_naive<<<grid, block>>>(d_in, d_out, W, H);
	CHECK(cudaEventRecord(e));
	CHECK(cudaEventSynchronize(e));
	float ms;
	CHECK(cudaEventElapsedTime(&ms, s, e));
	CHECK(cudaEventDestroy(s));
	CHECK(cudaEventDestroy(e));
	return ms;
}

double benchmark_smem(const float* d_in, float* d_out, const int W, const int H, const int iters) {
	constexpr auto BX = 16, BY = 16;
	dim3 block(BX, BY);
	dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY);

	cudaEvent_t s, e;
	CHECK(cudaEventCreate(&s));
	CHECK(cudaEventCreate(&e));
	CHECK(cudaEventRecord(s));
	for (auto i = 0; i < iters; ++i)
		blur3x3_smem<BX, BY><<<grid, block>>>(d_in, d_out, W, H);
	CHECK(cudaEventRecord(e));
	CHECK(cudaEventSynchronize(e));
	float ms;
	CHECK(cudaEventElapsedTime(&ms, s, e));
	CHECK(cudaEventDestroy(s));
	CHECK(cudaEventDestroy(e));
	return ms;
}

double benchmark_smem_reg(const float* d_in, float* d_out, const int W, const int H, const int iters) {
	constexpr auto BX = 16, BY = 16;
	constexpr auto tileW = BX * 2UL;
	dim3 block(BX, BY);
	dim3 grid((W + tileW - 1) / tileW, (H + BY - 1) / BY);

	constexpr auto sW = tileW + 2UL;
	constexpr auto sH = BY + 2UL;
	auto shared_bytes = sW * sH * sizeof(float);

	cudaEvent_t s, e;
	CHECK(cudaEventCreate(&s));
	CHECK(cudaEventCreate(&e));
	CHECK(cudaEventRecord(s));
	for (auto i = 0; i < iters; ++i)
		blur3x3_smem_reg<BX, BY><<<grid, block, shared_bytes>>>(d_in, d_out, W, H);
	CHECK(cudaEventRecord(e));
	CHECK(cudaEventSynchronize(e));
	float ms;
	CHECK(cudaEventElapsedTime(&ms, s, e));
	CHECK(cudaEventDestroy(s));
	CHECK(cudaEventDestroy(e));
	return ms;
}

bool compare_device_to_host(const float* d_res, const float* host_ref, const int W, const int H) {
	const auto N = static_cast<size_t>(W) * H;
	const auto tmp = static_cast<float*>(malloc(N * sizeof(float)));
	CHECK(cudaMemcpy(tmp, d_res, N * sizeof(float), cudaMemcpyDeviceToHost));
	auto ok = true;
	for (auto i = 0UL; i < N; ++i) {
		if (const auto a = tmp[i], b = host_ref[i]; fabs(a - b) > 1e-5f) {
			ok = false;
			// print first mismatch for debugging
			const auto y = i / W, x = i % W;
			printf("mismatch at (%zu,%zu): gpu=%.6f host=%.6f\n", x, y, a, b);
			break;
		}
	}
	free(tmp);
	return ok;
}

// CPU reference (single-threaded) for correctness verification
void cpu_blur_reference(const float* in, float* out, const int W, const int H) {
	for (auto y = 0; y < H; ++y) {
		for (auto x = 0; x < W; ++x) {
			auto sum = 0.0f;
			for (auto ky = -1; ky <= 1; ++ky)
				for (auto kx = -1, yy = min(max(y + ky, 0), H - 1); kx <= 1; ++kx)
					sum += in[yy * W + min(max(x + kx, 0), W - 1)] * KERNEL_3x3[ky + 1][kx + 1];
			out[y * W + x] = sum;
		}
	}
}

int main(const int argc, char* argv[]) {
	auto W = 4096L, H = 4096L;
	if (argc >= 3) {
		W = strtol(argv[1], nullptr, 10);
		H = strtol(argv[2], nullptr, 10);
	}
	auto iters = 20L;
	if (argc >= 4)
		iters = strtol(argv[3], nullptr, 10);
	printf("Image %ld x %ld, iterations %ld\n", W, H, iters);
	const auto N = static_cast<size_t>(W) * H;
	const auto bytes = N * sizeof(float);
	const auto h_in = static_cast<float*>(malloc(bytes));
	const auto h_ref = static_cast<float*>(malloc(bytes));
	init_image(h_in, W, H);
	cpu_blur_reference(h_in, h_ref, W, H);
	float *d_in, *d_out;
	CHECK(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
	CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes));
	CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
	// Warmup single run to avoid startup costs
	{
		dim3 b(16, 16), g((W + 15) / 16, (H + 15) / 16);
		blur3x3_naive<<<g, b>>>(d_in, d_out, W, H);
		CHECK(cudaDeviceSynchronize());
	}
	// Benchmark naive
	const auto ms_naive = benchmark_naive(d_in, d_out, W, H, iters);
	CHECK(cudaDeviceSynchronize());
	const auto ok_naive = compare_device_to_host(d_out, h_ref, W, H);
	printf("Naive: %.3f ms (avg), %s\n", ms_naive / iters, ok_naive ? "OK" : "WRONG");
	// Benchmark shared-memory
	const auto ms_smem = benchmark_smem(d_in, d_out, W, H, iters);
	CHECK(cudaDeviceSynchronize());
	const auto ok_smem = compare_device_to_host(d_out, h_ref, W, H);
	printf("SMEM:  %.3f ms (avg), %s\n", ms_smem / iters, ok_smem ? "OK" : "WRONG");
	// Benchmark shared-memory + register (2 outputs per thread)
	const auto ms_smem_reg = benchmark_smem_reg(d_in, d_out, W, H, iters);
	CHECK(cudaDeviceSynchronize());
	const auto ok_smem_reg = compare_device_to_host(d_out, h_ref, W, H);
	printf("SMEM+REG: %.3f ms (avg), %s\n", ms_smem_reg / iters, ok_smem_reg ? "OK" : "WRONG");
	// Print simple bandwidth numbers (approx): each output reads 9 reads but shared versions reduce global reads.
	const auto avg_naive = ms_naive / iters;
	const auto avg_smem = ms_smem / iters;
	const auto avg_smemr = ms_smem_reg / iters;
	const auto GB = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
	printf("\nAverage times (ms): naive=%.3f, smem=%.3f, smem_reg=%.3f\n", avg_naive, avg_smem, avg_smemr);
	printf("Image size: %.3f GB\n", GB);
	// Note: these â€œbandwidthâ€ numbers are very approximate
	printf("Bandwidth (approx): naive: %.2f GB/s, smem: %.2f GB/s, smem_reg: %.2f GB/s\n", GB / (avg_naive / 1000.0),
		   GB / (avg_smem / 1000.0), GB / (avg_smemr / 1000.0));
	// cleanup
	CHECK(cudaFree(d_in));
	CHECK(cudaFree(d_out));
	free(h_in);
	free(h_ref);
	return 0;
}
