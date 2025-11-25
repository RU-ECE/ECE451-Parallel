// a is global memory
__global__ void add(const int* a, const int* b, int* c) {
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	c[tid] = a[tid] + b[tid];
	int x; // this should be in a register
	__shared__ int y[32]; // shared memory (99kB?) L1 cache 2nd fastest
}
