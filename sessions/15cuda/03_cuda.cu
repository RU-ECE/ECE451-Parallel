// a is global memory
__global__ void add(int* a, int* b, int* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    c[tid] = a[tid] + b[tid];
    int x; // this should be in a register
    __shared__ int y[32]; // shared memory (99kB?) L1 cache 2nd fastest


}