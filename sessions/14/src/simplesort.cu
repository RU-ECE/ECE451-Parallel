#include <iostream>
#include <random>

#include "benchmark.hpp"

using namespace std;

/*
 * Progressive sorting demo
 * Author: Dov Kruger
 * quicksort on the CPU
 * mergesort brute force on the GPU
 * mergesort trying to do more exotic optimizations
 */


typedef unsigned int u32;

struct rec {
	u32 key;
	u32 recid;
};

/*
	Using standard C++ random, generate high quality random order of random
*/

void gen_sequential(rec x[], const u32 n) {
	for (u32 i = 0; i < n; i++) {
		x[i].key = i;
		x[i].recid = n - i;
	}
}

mt19937 gen(0);

/*
	host-side high quality Fischer-Yates shuffle of the array
*/
void shuffle(rec x[], const u32 n) {
	for (u32 i = n - 1; i > 0; i--) {
		uniform_int_distribution<u32> dist(0, i);
		const u32 r = dist(gen);
		const rec tmp = x[r];
		x[r] = x[i];
		x[i] = tmp;
	}
}

/*
	Quicksort using Lomuto partition scheme
*/
void quicksort(rec x[], const u32 L, const u32 R) {
	const u32 pivot = x[R].key;
	u32 i = L;

	for (u32 j = L; j < R; j++) {
		if (x[j].key < pivot) {
			const rec tmp = x[i];
			x[i] = x[j];
			x[j] = tmp;
			i++;
		}
	}
	const rec tmp = x[i];
	x[i] = x[R];
	x[R] = tmp;

	if (i > L)
		quicksort(x, L, i - 1);
	if (i + 1 < R)
		quicksort(x, i + 1, R);
}

void quicksort_wrapper(void* ptr, const unsigned int n) {
	const auto x = static_cast<rec*>(ptr);
	quicksort(x, 0, n - 1);
}

/*
	Given an array x[] of size n elements with sorted groups of size sorted_size, merge each pair of
	2 sorted groups into one in the output array y[].
	The result is an array with groups of size 2*sorted_size.
*/
__global__ void merge(rec x[], rec y[], const u32 n, const u32 sorted_size) {
	const u32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_id >= n / (2 * sorted_size))
		return;

	const u32 left_start = thread_id * 2 * sorted_size;
	const u32 right_start = left_start + sorted_size;
	const u32 output_start = left_start;

	const u32 left_end = right_start;
	const u32 right_end = right_start + sorted_size < n ? right_start + sorted_size : n;

	if (right_start >= n) {
		// Only left side exists, just copy it
		for (u32 i = left_start; i < n && i < left_end; i++)
			y[i] = x[i];
		return;
	}

	u32 i = left_start, j = right_start, k = output_start;
	while (i < left_end && j < right_end)
		y[k++] = x[i].key < x[j].key ? x[i++] : x[j++];
	while (i < left_end)
		y[k++] = x[i++];
	while (j < right_end)
		y[k++] = x[j++];
}


/*
	This is a primitive merge sort. It is doing a lot of reading and writing to global memory.
	Also, it does not end early.With a recursive merge, if you notice that left[n] < right[0], you can stop.
	Here we have to copy every time.
	NOTE: Kernel launches must be done from host, not from within a kernel.
*/

__forceinline__ __device__ void sort2(rec& a, rec& b) {
	if (a.key > b.key) {
		const rec tmp = a;
		a = b;
		b = tmp;
	}
}

/*
	sortingnetwork8 is an optimal sorting network that will read each group of 8 elements into registers, sort them,
	and write them back. This is much more efficient than the first few passes of merge sort.
	Once we see how this works, we can do this in bigger groups (16, 32, 64). We could also try to have multiple threads
   working on the same optimal sorting network.
*/
__global__ void sortingnetwork8(rec* x, rec* y, const u32 n) {
	const u32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_id >= n / 8)
		return; // we only need this many threads

	rec a = x[thread_id * 8];
	rec b = x[thread_id * 8 + 1];
	rec c = x[thread_id * 8 + 2];
	rec d = x[thread_id * 8 + 3];
	rec e = x[thread_id * 8 + 4];
	rec f = x[thread_id * 8 + 5];
	rec g = x[thread_id * 8 + 6];
	rec h = x[thread_id * 8 + 7];
	/*
		[(0,2),(1,3),(4,6),(5,7)]
		[(0,4),(1,5),(2,6),(3,7)]
		[(0,1),(2,3),(4,5),(6,7)]
		[(2,4),(3,5)]
		[(1,4),(3,6)]
		[(1,2),(3,4),(5,6)]
	*/
	sort2(a, c);
	sort2(e, g);
	sort2(b, d);
	sort2(f, h);
	sort2(a, e);
	sort2(b, f);
	sort2(c, g);
	sort2(d, h);
	sort2(b, c);
	sort2(f, g);
	sort2(a, b);
	sort2(c, d);
	sort2(e, f);
	sort2(g, h);
	sort2(c, e);
	sort2(d, f);
	sort2(b, c);
	sort2(d, e);
	sort2(f, g);

	x[thread_id * 8] = a;
	x[thread_id * 8 + 1] = b;
	x[thread_id * 8 + 2] = c;
	x[thread_id * 8 + 3] = d;
	x[thread_id * 8 + 4] = e;
	x[thread_id * 8 + 5] = f;
	x[thread_id * 8 + 6] = g;
	x[thread_id * 8 + 7] = h;
}

/*
	Given an array x[] of size n elements with sorted groups of size sorted_size, merge each 4 groups
	into one in the output array y[].
	The result is an  array with groups of size 4*sorted_size.
	This skips one pass of merge sort, and more importantly, we are going to try to use local registers to merge.
	The if statements are still going to diverge though.
	At the moment, local blocks of size 8, but 16 is probably optimal because CUDA uses
	blocks of 32-32bit values for cache line. We are only using 16.
*/
__global__ void merge4(rec x[], rec y[], const u32 n, const u32 sorted_size) {
	const u32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_id >= n / (4 * sorted_size))
		return;

	const u32 g0 = thread_id * 4 * sorted_size;
	const u32 g1 = g0 + sorted_size;
	const u32 g2 = g1 + sorted_size;
	const u32 g3 = g2 + sorted_size;
	const u32 g_end = g3 + sorted_size < n ? g3 + sorted_size : n;

	u32 i = g0, j = g1, k = g2, m = g3, out = g0;

	const u32 i_end = g1;
	const u32 j_end = g2;
	const u32 k_end = g3;
	const u32 m_end = g_end;

	// 4-way merge from global memory
	while (i < i_end && j < j_end && k < k_end && m < m_end) {
		u32 min_key = x[i].key;
		u32 min_idx = 0;
		if (x[j].key < min_key) {
			min_key = x[j].key;
			min_idx = 1;
		}
		if (x[k].key < min_key) {
			min_key = x[k].key;
			min_idx = 2;
		}
		if (x[m].key < min_key) {
			min_key = x[m].key;
			min_idx = 3;
		}
		if (min_idx == 0)
			y[out++] = x[i++];
		else if (min_idx == 1)
			y[out++] = x[j++];
		else if (min_idx == 2)
			y[out++] = x[k++];
		else
			y[out++] = x[m++];
	}
	// finish remaining elements
	while (i < i_end && j < j_end && k < k_end) {
		if (x[i].key < x[j].key)
			if (x[i].key < x[k].key)
				y[out++] = x[i++];
			else
				y[out++] = x[k++];
		else if (x[j].key < x[k].key)
			y[out++] = x[j++];
		else
			y[out++] = x[k++];
	}
	while (i < i_end && j < j_end && m < m_end) {
		if (x[i].key < x[j].key)
			if (x[i].key < x[m].key)
				y[out++] = x[i++];
			else
				y[out++] = x[m++];
		else if (x[j].key < x[m].key)
			y[out++] = x[j++];
		else
			y[out++] = x[m++];
	}
	while (i < i_end && k < k_end && m < m_end) {
		if (x[i].key < x[k].key)
			if (x[i].key < x[m].key)
				y[out++] = x[i++];
			else
				y[out++] = x[m++];
		else if (x[k].key < x[m].key)
			y[out++] = x[k++];
		else
			y[out++] = x[m++];
	}
	while (j < j_end && k < k_end && m < m_end) {
		if (x[j].key < x[k].key)
			if (x[j].key < x[m].key)
				y[out++] = x[j++];
			else
				y[out++] = x[m++];
		else if (x[k].key < x[m].key)
			y[out++] = x[k++];
		else
			y[out++] = x[m++];
	}
	while (i < i_end && j < j_end)
		y[out++] = x[i].key < x[j].key ? x[i++] : x[j++];
	while (i < i_end && k < k_end)
		y[out++] = x[i].key < x[k].key ? x[i++] : x[k++];
	while (i < i_end && m < m_end)
		y[out++] = x[i].key < x[m].key ? x[i++] : x[m++];
	while (j < j_end && k < k_end)
		y[out++] = x[j].key < x[k].key ? x[j++] : x[k++];
	while (j < j_end && m < m_end)
		y[out++] = x[j].key < x[m].key ? x[j++] : x[m++];
	while (k < k_end && m < m_end)
		y[out++] = x[k].key < x[m].key ? x[k++] : x[m++];
	while (i < i_end)
		y[out++] = x[i++];
	while (j < j_end)
		y[out++] = x[j++];
	while (k < k_end)
		y[out++] = x[k++];
	while (m < m_end)
		y[out++] = x[m++];
}


/*
	This is an attempt at a more efficient merge sort.
	We are going to copy into local registers, merge 4 ways (skipping one generation) then copy back to global memory.
	NOTE: Kernel launches must be done from host, not from within a kernel.
*/

__global__ void initial_sort(rec x[], const u32 n, const u32 block_size) {
	const u32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	const u32 start = thread_id * block_size;
	if (start >= n)
		return;
	const u32 end = start + block_size < n ? start + block_size : n;

	// Simple bubble sort variant for small blocks (better for GPU than insertion sort)
	for (u32 i = start; i < end - 1; i++) {
		for (u32 j = start; j < end - 1 - (i - start); j++) {
			if (x[j].key > x[j + 1].key) {
				const rec tmp = x[j];
				x[j] = x[j + 1];
				x[j + 1] = tmp;
			}
		}
	}
}

void merge_sort_wrapper(void* ptr, const unsigned int n) {
	const auto x_gpu = static_cast<rec*>(ptr);
	extern rec* y_gpu_global;
	extern u32 num_threads_global;

	// First, sort initial blocks to reduce number of merge passes
	constexpr u32 initial_block_size = 256;
	const u32 num_initial_blocks = (n + initial_block_size - 1) / initial_block_size;
	u32 num_blocks_initial = (num_initial_blocks + num_threads_global - 1) / num_threads_global;
	if (num_blocks_initial == 0)
		num_blocks_initial = 1;
	initial_sort<<<num_blocks_initial, num_threads_global>>>(x_gpu, n, initial_block_size);

	rec* a = x_gpu;
	rec* b = y_gpu_global;
	for (u32 sorted_size = initial_block_size; sorted_size < n; sorted_size *= 2) {
		const u32 num_pairs = n / (2 * sorted_size);
		if (num_pairs == 0)
			break;
		u32 num_blocks = (num_pairs + num_threads_global - 1) / num_threads_global;
		if (num_blocks == 0)
			num_blocks = 1;
		merge<<<num_blocks, num_threads_global>>>(a, b, n, sorted_size);
		rec* tmp = a;
		a = b;
		b = tmp;
	}
	// Only sync once at the end, not after every pass
	cudaDeviceSynchronize();
	if (a != x_gpu)
		cudaMemcpy(x_gpu, a, n * sizeof(rec), cudaMemcpyDeviceToDevice);
}

void merge_sort2_wrapper(void* ptr, const unsigned int n) {
	const auto x_gpu = static_cast<rec*>(ptr);
	extern rec* y_gpu_global;
	extern u32 num_threads_global;
	if (const u32 num_groups = n / 8; num_groups > 0) {
		u32 num_blocks_sort = (num_groups + num_threads_global - 1) / num_threads_global;
		if (num_blocks_sort == 0)
			num_blocks_sort = 1;
		sortingnetwork8<<<num_blocks_sort, num_threads_global>>>(x_gpu, y_gpu_global, n);
		cudaDeviceSynchronize();
	}
	rec* a = x_gpu;
	rec* b = y_gpu_global;
	for (u32 sorted_size = 8; sorted_size < n; sorted_size *= 4) {
		const u32 num_quads = n / (4 * sorted_size);
		if (num_quads == 0)
			break;
		u32 num_blocks_merge4 = (num_quads + num_threads_global - 1) / num_threads_global;
		if (num_blocks_merge4 == 0)
			num_blocks_merge4 = 1;
		merge4<<<num_blocks_merge4, num_threads_global>>>(a, b, n, sorted_size);
		rec* tmp = a;
		a = b;
		b = tmp;
	}
	if (a != x_gpu)
		cudaMemcpy(x_gpu, y_gpu_global, n * sizeof(rec), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
}

#if 0
// forget this
__global__ void copySlice(const u32 inputArray[], int startIndex, int length) {
	// Allocate shared memory for the local array
	__shared__ int localArray[1024]; // Assuming a maximum slice length of 256

	// Calculate the global index for this thread
	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

	// Calculate the local index for this thread
	int localIndex = threadIdx.x;

	// Copy the slice from global memory to shared memory
	if (globalIndex >= startIndex && globalIndex < startIndex + length)
		localArray[localIndex] = inputArray[globalIndex];

	// Wait for all threads to finish copying to shared memory
	__syncthreads();

	// Copy the local array to output memory
	if (globalIndex >= startIndex && globalIndex < startIndex + length)
		outputArray[globalIndex] = localArray[localIndex];
}
#endif

void check_sorted(const rec* x, const u32 n) {
	u32 error = 0;
	vector<u32> errors;
	for (u32 i = 0; i < n; i++) {
		if (x[i].key != i) {
			error++;
			errors.push_back(i);
		}
	}
	const u32 max_errors = error > 10 ? 10 : error;
	if (error > 0) {
		cout << "Errors: " << error << endl;

		for (u32 i = 0; i < max_errors; i++)
			cout << errors[i] << " ";
		cout << endl;
	}
}


/**
	Generate an array from 0 to n-1. Randomly shuffle it.
	Copy to preserve the master copy
	Benchmark quicksort on the CPU. then make sure the original values are there.
	Benchmark mergesort using a brute-force, not particularly efficient threading on the GPU
	copy the results back and make sure the original values are there.
	Benchmark mergesort2 using a more efficient threading on the GPU
	copy the results back and make sure the original values are there.
*/
constexpr u32 n = 16 * 1024 * 1024; // 64M elements, 512MB data (each element is 8 bytes)
rec* y_gpu_global;
u32 num_threads_global;

int main() {
	int deviceCount;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if (err != cudaSuccess) {
		cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << endl;
		return 1;
	}
	if (deviceCount == 0) {
		cerr << "No CUDA devices found" << endl;
		return 1;
	}
	cout << "Found " << deviceCount << " CUDA device(s)" << endl;

	err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << endl;
		return 1;
	}

	// Clear any previous errors
	cudaGetLastError();

	cudaDeviceProp prop;
	err = cudaGetDeviceProperties(&prop, 0);
	if (err == cudaSuccess) {
		cout << "Using device: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << endl;
		cout << "Total memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << endl;
	}

	// Force context creation with a dummy allocation
	void* dummy;
	err = cudaMalloc(&dummy, 1);
	if (err != cudaSuccess) {
		cerr << "Failed to create CUDA context: " << cudaGetErrorString(err) << endl;
		return 1;
	}
	cudaFree(dummy);

	auto orig = new rec[n];
	gen_sequential(orig, n); // fill with sequential keys
	shuffle(orig, n); // shuffle the array into a random order, but we know what the data should be
	auto x = new rec[n];
	memcpy(x, orig, n * sizeof(rec));

	benchmark("quicksort", quicksort_wrapper, x, n); // sort the array on the CPU
	check_sorted(x, n); // check that the array is sorted back into sequential order

	// copy the array to the GPU
	rec* x_gpu;
	rec* y_gpu;
	size_t bytes = n * sizeof(rec);
	cout << "Allocating " << (bytes / (1024 * 1024)) << " MB on GPU for x_gpu" << endl;
	err = cudaMalloc(reinterpret_cast<void**>(&x_gpu), bytes);
	if (err != cudaSuccess) {
		cerr << "cudaMalloc x_gpu failed: " << cudaGetErrorString(err) << " (code: " << err << ")" << endl;
		return 1;
	}
	cout << "Allocating " << (bytes / (1024 * 1024)) << " MB on GPU for y_gpu" << endl;
	err = cudaMalloc(reinterpret_cast<void**>(&y_gpu), bytes);
	if (err != cudaSuccess) {
		cerr << "cudaMalloc y_gpu failed: " << cudaGetErrorString(err) << endl;
		cudaFree(x_gpu);
		return 1;
	}
	y_gpu_global = y_gpu;
	num_threads_global = 256;

	// Reset x from original shuffled data
	memcpy(x, orig, n * sizeof(rec));
	cudaMemcpy(x_gpu, x, n * sizeof(rec), cudaMemcpyHostToDevice);

	benchmark("merge_sort", merge_sort_wrapper, x_gpu, n);
	cudaMemcpy(x, x_gpu, n * sizeof(rec), cudaMemcpyDeviceToHost);
	check_sorted(x, n); // check that the array is sorted back into sequential order

	// Reset x from original shuffled data
	memcpy(x, orig, n * sizeof(rec));
	cudaMemcpy(x_gpu, x, n * sizeof(rec), cudaMemcpyHostToDevice);

	benchmark("merge_sort2", merge_sort2_wrapper, x_gpu, n);
	cudaMemcpy(x, x_gpu, n * sizeof(rec), cudaMemcpyDeviceToHost);
	check_sorted(x, n); // check that the array is sorted back into sequential order

	cudaFree(y_gpu);

	cudaFree(x_gpu);
	delete[] x;
	delete[] orig;
}
