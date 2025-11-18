// blur3x3_bench.cu
// Compile: nvcc -O3 blur3x3_bench.cu -o blur3x3_bench
// Run: ./blur3x3_bench [width] [height] [iters]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        fprintf(stderr,"CUDA error %s:%d: %s\n",         \
                __FILE__,__LINE__,cudaGetErrorString(err)); \
        exit(1);                                         \
    }                                                    \
} while(0)

static const float KERNEL_3x3[3][3] = {
    {1/9.f, 1/9.f, 1/9.f},
    {1/9.f, 1/9.f, 1/9.f},
    {1/9.f, 1/9.f, 1/9.f}
};

/*
    P  P  P 
    P  C  P
    P  P  P
    C is the center pixel

    1 2 3 4           1+2+3+2+3+5+3+5+8 = 32 / 9 = 3.55
    2 3 5 6           2+3+5+3+5+8+5+8+9 = 48 / 9 = 5.333333333333333
    3 5 8 9
    4 6 5 1


     3.55 
*/


// -----------------------------
// Kernel 1: Naive brute-force
// Each thread computes one output pixel reading 9 global loads
// -----------------------------
__global__
void blur3x3_naive(const float* __restrict__ in, float* __restrict__ out, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float sum = 0.0f;
    // 3x3 kernel centered on (x,y). Use clamp at borders.
    for (int ky = -1; ky <= 1; ++ky) {
        int yy = min(max(y + ky, 0), H-1);
        for (int kx = -1; kx <= 1; ++kx) {
            int xx = min(max(x + kx, 0), W-1);
            sum += in[yy * W + xx] * KERNEL_3x3[ky+1][kx+1];
        }
    }
    out[y * W + x] = sum;
}

// -----------------------------
// Kernel 2: Shared-memory tiling (one output per thread)
// - Each block loads tile of size (BX x BY) plus 1-pixel halo around
// - Each thread loads its own interior pixel and boundary threads load halo
// -----------------------------
template<int BX, int BY>
__global__
void blur3x3_smem(const float* __restrict__ in, float* __restrict__ out, int W, int H)
{
    // shared tile dimensions: BX+2 by BY+2
    __shared__ float s[(BY+2)*(BX+2)];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BX + tx;
    int y = blockIdx.y * BY + ty;

    // shared memory coordinates (shifted by +1 for halo)
    int s_x = tx + 1;
    int s_y = ty + 1;
    int s_idx = s_y * (BX+2) + s_x;

    // load center element (if within image) else load clamped border
    int xc = min(max(x, 0), W-1);
    int yc = min(max(y, 0), H-1);
    s[s_idx] = in[yc * W + xc];

    // load halos: threads on edges load the surrounding pixels
    // left
    if (tx == 0) {
        int xl = min(max(x-1, 0), W-1);
        s[s_y * (BX+2) + (s_x - 1)] = in[yc * W + xl];
    }
    // right
    if (tx == BX-1) {
        int xr = min(max(x+1, 0), W-1);
        s[s_y * (BX+2) + (s_x + 1)] = in[yc * W + xr];
    }
    // top
    if (ty == 0) {
        int yt = min(max(y-1, 0), H-1);
        s[(s_y - 1) * (BX+2) + s_x] = in[yt * W + xc];
    }
    // bottom
    if (ty == BY-1) {
        int yb = min(max(y+1, 0), H-1);
        s[(s_y + 1) * (BX+2) + s_x] = in[yb * W + xc];
    }
    // corners (four)
    if (tx == 0 && ty == 0) {
        int xl = min(max(x-1, 0), W-1);
        int yt = min(max(y-1, 0), H-1);
        s[(s_y-1) * (BX+2) + (s_x-1)] = in[yt * W + xl];
    }
    if (tx == BX-1 && ty == 0) {
        int xr = min(max(x+1, 0), W-1);
        int yt = min(max(y-1, 0), H-1);
        s[(s_y-1) * (BX+2) + (s_x+1)] = in[yt * W + xr];
    }
    if (tx == 0 && ty == BY-1) {
        int xl = min(max(x-1, 0), W-1);
        int yb = min(max(y+1, 0), H-1);
        s[(s_y+1) * (BX+2) + (s_x-1)] = in[yb * W + xl];
    }
    if (tx == BX-1 && ty == BY-1) {
        int xr = min(max(x+1, 0), W-1);
        int yb = min(max(y+1, 0), H-1);
        s[(s_y+1) * (BX+2) + (s_x+1)] = in[yb * W + xr];
    }

    __syncthreads();

    // compute output if within image bounds
    if (x < W && y < H) {
        float sum = 0.0f;
        // unrolled 3x3 loop reading from shared memory
        sum += s[(s_y-1)*(BX+2) + (s_x-1)] * KERNEL_3x3[0][0];
        sum += s[(s_y-1)*(BX+2) + (s_x  )] * KERNEL_3x3[0][1];
        sum += s[(s_y-1)*(BX+2) + (s_x+1)] * KERNEL_3x3[0][2];

        sum += s[(s_y  )*(BX+2) + (s_x-1)] * KERNEL_3x3[1][0];
        sum += s[(s_y  )*(BX+2) + (s_x  )] * KERNEL_3x3[1][1];
        sum += s[(s_y  )*(BX+2) + (s_x+1)] * KERNEL_3x3[1][2];

        sum += s[(s_y+1)*(BX+2) + (s_x-1)] * KERNEL_3x3[2][0];
        sum += s[(s_y+1)*(BX+2) + (s_x  )] * KERNEL_3x3[2][1];
        sum += s[(s_y+1)*(BX+2) + (s_x+1)] * KERNEL_3x3[2][2];

        out[y * W + x] = sum;
    }
}

// -----------------------------
// Kernel 3: Shared-memory + register blocking
// Each thread computes TWO horizontal outputs (when possible).
// Grid.x is sized so each block handles a tile width = BX*2.
// Shared memory width = BX*2 + 2
// -----------------------------
template<int BX, int BY>
__global__
void blur3x3_smem_reg(const float* __restrict__ in, float* __restrict__ out, int W, int H)
{
    // tile width handled by block = BX*2, tile height = BY
    const int tileW = BX*2;
    const int tileH = BY;
    const int sW = tileW + 2;
    const int sH = tileH + 2;
    extern __shared__ float s_ext[]; // size sW * sH
    float* s = s_ext;

    int tx = threadIdx.x; // 0..BX-1
    int ty = threadIdx.y; // 0..BY-1

    // base coordinates for block
    int baseX = blockIdx.x * tileW;
    int baseY = blockIdx.y * tileH;

    // each thread will load two interior pixels (x0,x1) into shared memory
    int x0 = baseX + tx;
    int x1 = baseX + tx + BX; // second column handled by same thread
    int y = baseY + ty;

    // shared memory coords: shift by +1 (halo)
    int s_x0 = tx + 1;
    int s_x1 = tx + 1 + BX;
    int s_y  = ty + 1;

    // load center pixels (clamped)
    int yc = min(max(y, 0), H-1);

    // load center two pixels (if they exist)
    if (x0 < W) {
        int xc0 = min(max(x0, 0), W-1);
        s[s_y * sW + s_x0] = in[yc * W + xc0];
    } else {
        s[s_y * sW + s_x0] = 0.0f;
    }
    if (x1 < W) {
        int xc1 = min(max(x1, 0), W-1);
        s[s_y * sW + s_x1] = in[yc * W + xc1];
    } else {
        s[s_y * sW + s_x1] = 0.0f;
    }

    // load left/right halo columns: threads with tx==0 load left halo column (two spots)
    if (tx == 0) {
        int xl = min(max(baseX - 1, 0), W-1);
        int xr = min(max(baseX + tileW - 1, 0), W-1);
        // left halo (at s_x0 -1)
        s[s_y * sW + 0] = in[yc * W + xl];
        // rightmost interior location was loaded by thread tx=BX-1's x1 slot; but we should ensure right halo loaded at s_x = tileW+1
        s[s_y * sW + (sW-1)] = in[yc * W + xr];
    }

    // load top/bottom halo rows (for this thread's two x positions)
    if (ty == 0) {
        int yt = min(max(y-1, 0), H-1);
        // top for x0
        if (x0 < W) {
            int xc0 = min(max(x0, 0), W-1);
            s[(s_y - 1) * sW + s_x0] = in[yt * W + xc0];
        } else {
            s[(s_y - 1) * sW + s_x0] = 0.0f;
        }
        // top for x1
        if (x1 < W) {
            int xc1 = min(max(x1, 0), W-1);
            s[(s_y - 1) * sW + s_x1] = in[yt * W + xc1];
        } else {
            s[(s_y - 1) * sW + s_x1] = 0.0f;
        }
        // corners: left-top and right-top (for tx==0 thread)
        if (tx == 0) {
            int xl = min(max(baseX - 1, 0), W-1);
            s[(s_y - 1) * sW + 0] = in[yt * W + xl];
            int xr = min(max(baseX + tileW - 1, 0), W-1);
            s[(s_y - 1) * sW + (sW-1)] = in[yt * W + xr];
        }
    }
    if (ty == BY-1) {
        int yb = min(max(y+1, 0), H-1);
        if (x0 < W) {
            int xc0 = min(max(x0, 0), W-1);
            s[(s_y + 1) * sW + s_x0] = in[yb * W + xc0];
        } else {
            s[(s_y + 1) * sW + s_x0] = 0.0f;
        }
        if (x1 < W) {
            int xc1 = min(max(x1, 0), W-1);
            s[(s_y + 1) * sW + s_x1] = in[yb * W + xc1];
        } else {
            s[(s_y + 1) * sW + s_x1] = 0.0f;
        }
        if (tx == 0) {
            int xl = min(max(baseX - 1, 0), W-1);
            s[(s_y + 1) * sW + 0] = in[yb * W + xl];
            int xr = min(max(baseX + tileW - 1, 0), W-1);
            s[(s_y + 1) * sW + (sW-1)] = in[yb * W + xr];
        }
    }

    __syncthreads();

    // Now each thread computes two outputs (for x0 and x1) using registers
    if (x0 < W && y < H) {
        float a00 = s[(s_y-1)*sW + (s_x0-1)];
        float a01 = s[(s_y-1)*sW + (s_x0  )];
        float a02 = s[(s_y-1)*sW + (s_x0+1)];

        float a10 = s[(s_y  )*sW + (s_x0-1)];
        float a11 = s[(s_y  )*sW + (s_x0  )];
        float a12 = s[(s_y  )*sW + (s_x0+1)];

        float a20 = s[(s_y+1)*sW + (s_x0-1)];
        float a21 = s[(s_y+1)*sW + (s_x0  )];
        float a22 = s[(s_y+1)*sW + (s_x0+1)];

        float sum0 = a00*KERNEL_3x3[0][0] + a01*KERNEL_3x3[0][1] + a02*KERNEL_3x3[0][2]
                   + a10*KERNEL_3x3[1][0] + a11*KERNEL_3x3[1][1] + a12*KERNEL_3x3[1][2]
                   + a20*KERNEL_3x3[2][0] + a21*KERNEL_3x3[2][1] + a22*KERNEL_3x3[2][2];

        out[y * W + x0] = sum0;
    }

    // second output
    if (x1 < W && y < H) {
        float b00 = s[(s_y-1)*sW + (s_x1-1)];
        float b01 = s[(s_y-1)*sW + (s_x1  )];
        float b02 = s[(s_y-1)*sW + (s_x1+1)];

        float b10 = s[(s_y  )*sW + (s_x1-1)];
        float b11 = s[(s_y  )*sW + (s_x1  )];
        float b12 = s[(s_y  )*sW + (s_x1+1)];

        float b20 = s[(s_y+1)*sW + (s_x1-1)];
        float b21 = s[(s_y+1)*sW + (s_x1  )];
        float b22 = s[(s_y+1)*sW + (s_x1+1)];

        float sum1 = b00*KERNEL_3x3[0][0] + b01*KERNEL_3x3[0][1] + b02*KERNEL_3x3[0][2]
                   + b10*KERNEL_3x3[1][0] + b11*KERNEL_3x3[1][1] + b12*KERNEL_3x3[1][2]
                   + b20*KERNEL_3x3[2][0] + b21*KERNEL_3x3[2][1] + b22*KERNEL_3x3[2][2];

        out[y * W + x1] = sum1;
    }
}

// -----------------------------
// Host helpers: init, verify, benchmark
// -----------------------------
void init_image(float* a, int W, int H)
{
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            // something non-trivial
            a[y*W + x] = (float)(((x*37) ^ (y*17)) & 255) / 255.0f;
        }
}

double benchmark_naive(const float* d_in, float* d_out, int W, int H, int iters)
{
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1)/block.x, (H + block.y - 1)/block.y);

    cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) {
        blur3x3_naive<<<grid, block>>>(d_in, d_out, W, H);
    }
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    float ms; CHECK(cudaEventElapsedTime(&ms, s, e));
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    return ms;
}

double benchmark_smem(const float* d_in, float* d_out, int W, int H, int iters)
{
    const int BX = 16, BY = 16;
    dim3 block(BX, BY);
    dim3 grid((W + BX - 1)/BX, (H + BY - 1)/BY);

    cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) {
        blur3x3_smem<BX,BY><<<grid, block>>>(d_in, d_out, W, H);
    }
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    float ms; CHECK(cudaEventElapsedTime(&ms, s, e));
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    return ms;
}

double benchmark_smem_reg(const float* d_in, float* d_out, int W, int H, int iters)
{
    const int BX = 16, BY = 16;
    const int tileW = BX*2;
    dim3 block(BX, BY);
    dim3 grid((W + tileW - 1)/tileW, (H + BY - 1)/BY);

    size_t sW = (tileW + 2);
    size_t sH = (BY + 2);
    size_t shared_bytes = sW * sH * sizeof(float);

    cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) {
        blur3x3_smem_reg<BX,BY><<<grid, block, shared_bytes>>>(d_in, d_out, W, H);
    }
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    float ms; CHECK(cudaEventElapsedTime(&ms, s, e));
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    return ms;
}

bool compare_device_to_host(const float* d_res, const float* host_ref, int W, int H)
{
    size_t N = (size_t)W * H;
    float* tmp = (float*)malloc(N * sizeof(float));
    CHECK(cudaMemcpy(tmp, d_res, N*sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        float a = tmp[i], b = host_ref[i];
        if (fabs(a - b) > 1e-5f) {
            ok = false;
            // print first mismatch for debugging
            size_t y = i / W, x = i % W;
            printf("mismatch at (%zu,%zu): gpu=%.6f host=%.6f\n", x, y, a, b);
            break;
        }
    }
    free(tmp);
    return ok;
}

// CPU reference (single-threaded) for correctness verification
void cpu_blur_reference(const float* in, float* out, int W, int H)
{
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                int yy = std::min(std::max(y + ky, 0), H-1);
                for (int kx = -1; kx <= 1; ++kx) {
                    int xx = std::min(std::max(x + kx, 0), W-1);
                    sum += in[yy * W + xx] * KERNEL_3x3[ky+1][kx+1];
                }
            }
            out[y * W + x] = sum;
        }
    }
}

int main(int argc, char** argv)
{
    int W = 4096, H = 4096;
    if (argc >= 3) { W = atoi(argv[1]); H = atoi(argv[2]); }
    int iters = 20;
    if (argc >= 4) iters = atoi(argv[3]);
    printf("Image %d x %d, iterations %d\n", W, H, iters);

    size_t N = (size_t)W * H;
    size_t bytes = N * sizeof(float);

    float* h_in  = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    init_image(h_in, W, H);
    cpu_blur_reference(h_in, h_ref, W, H);

    float *d_in, *d_out;
    CHECK(cudaMalloc((void**)&d_in, bytes));
    CHECK(cudaMalloc((void**)&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Warmup single run to avoid startup costs
    {
        dim3 b(16,16), g((W+15)/16, (H+15)/16);
        blur3x3_naive<<<g,b>>>(d_in, d_out, W, H);
        CHECK(cudaDeviceSynchronize());
    }

    // Benchmark naive
    double ms_naive = benchmark_naive(d_in, d_out, W, H, iters);
    CHECK(cudaDeviceSynchronize());
    bool ok_naive = compare_device_to_host(d_out, h_ref, W, H);
    printf("Naive: %.3f ms (avg), %s\n", ms_naive / iters, ok_naive ? "OK" : "WRONG");

    // Benchmark shared-memory
    double ms_smem = benchmark_smem(d_in, d_out, W, H, iters);
    CHECK(cudaDeviceSynchronize());
    bool ok_smem = compare_device_to_host(d_out, h_ref, W, H);
    printf("SMEM:  %.3f ms (avg), %s\n", ms_smem / iters, ok_smem ? "OK" : "WRONG");

    // Benchmark shared-memory + register (2 outputs per thread)
    double ms_smem_reg = benchmark_smem_reg(d_in, d_out, W, H, iters);
    CHECK(cudaDeviceSynchronize());
    bool ok_smem_reg = compare_device_to_host(d_out, h_ref, W, H);
    printf("SMEM+REG: %.3f ms (avg), %s\n", ms_smem_reg / iters, ok_smem_reg ? "OK" : "WRONG");

    // Print simple bandwidth numbers (approx): each output reads 9 reads but shared versions reduce global reads.
    double avg_naive = ms_naive / iters;
    double avg_smem  = ms_smem  / iters;
    double avg_smemr = ms_smem_reg / iters;
    double GB = (double)bytes / (1024.0*1024.0*1024.0);

    printf("\nAverage times (ms): naive=%.3f, smem=%.3f, smem_reg=%.3f\n", avg_naive, avg_smem, avg_smemr);
    printf("Image size: %.3f GB\n", GB);
    // Note: these "bandwidth" numbers are very approximate
    printf("Bandwidth (approx): naive: %.2f GB/s, smem: %.2f GB/s, smem_reg: %.2f GB/s\n",
           GB / (avg_naive/1000.0), GB / (avg_smem/1000.0), GB / (avg_smemr/1000.0));

    // cleanup
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
    free(h_in); free(h_ref);

    return 0;
}
