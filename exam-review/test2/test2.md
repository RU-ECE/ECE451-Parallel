# Test 2 Review

## Topics

- OpenMP
- CUDA

## OpenMP

- OpenMP programming model
	- providing additional information to automatically parallelize code
	- optional features in compilers (works single threaded)
	- #pragma c++ feature

```c++
#pragma yak 5   // ignored if your compiler doesn't know the pragma
```

- OpenMP commands

```c++
#pragma omp parallel
#pragma omp critical 
#pragma omp parallel for // split the loop, default is into num_threads chunks
#pragma omp parallel for private(var) // var is local to the thread, high performance
#pragma omp parallel for shared(var) // var is shared, and therefore if you write, trouble!
#pragma omp parallel for reduction(op:var)
#pragma omp simd
#pragma omp barrier // NOT ON TEST
#pragma omp task    // NOT ON TEST
```

- `#pragma omp parallel`: Creates a parallel region where multiple threads execute the code block concurrently.
- `#pragma omp critical`: Ensures only one thread executes the code block at a time, providing mutual exclusion.
- `#pragma omp parallel for`: Combines parallel region creation with loop distribution, automatically dividing loop
  iterations among threads.
- `#pragma omp parallel for private(var)`: Each thread gets its own private copy of the variable, initialized to
  undefined value.
- `#pragma omp parallel for shared(var)`: All threads share the same variable, requiring synchronization for safe
  access.
- `#pragma omp parallel for firstprivate(var)`: Each thread gets a private copy initialized with the value from before
  the parallel region.
- `#pragma omp simd`: Enables SIMD vectorization for the loop, allowing multiple iterations to execute in parallel using
  vector instructions.
- `#pragma omp sections`: Defines a block containing sections that will be distributed among threads.
- `#pragma omp section`: Marks a section within a sections block to be executed by one thread.
- `#pragma omp single`: Ensures the code block is executed by only one thread (not necessarily the master).
- `#pragma omp master`: Ensures the code block is executed only by the master thread (thread 0).
- `#pragma omp barrier`: Synchronization point where all threads must wait until every thread reaches this point.
- `#pragma omp atomic`: Ensures atomic update of a memory location, preventing race conditions on simple operations.
- `#pragma omp reduction(op:var)`: Performs reduction operation (e.g., +, *, max) on variable, combining results from
  all threads.
- `#pragma omp parallel sections`: Combines parallel region creation with sections, distributing sections among threads.
- `#pragma omp task`: Defines an independent task that can be executed by any available thread, enabling task
  parallelism.

Example of omp parallel for share()
Why we want to share a variable

```c++
int a[1000], b[1000], c[1000];
// Arrays are shared by default - all threads write to different elements
#pragma omp parallel for shared(a, b, c)
for (int i = 0; i < 1000; i++) {
    c[i] = a[i] * b[i];  // Each thread works on different i, no conflict
}
```

1. Where is the best place to put omp parallel for
2. Where is the best place to put omp simd
3. Why is the combination so good? Aren't we maxing out ram?

Example of omp parallel for on the outside with omp simd on the inside loop

```c++
void matmult(const double* A, const double* B, double* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            #pragma omp simd
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}
```

Example of omp parallel for on the outside with omp simd on the inside loop

```c++
void matmult(const double* A, const double* B, double* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n+j] = 0.0;
            #pragma omp simd
            for (int k = 0; k < n; k++) {
                C[i*n+j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
```

The performance increase to be expected from n cpus depends on the ratio of CPU usage to memory
Ahmdal's law

1 cpu = 100%
2 cpu = 80%
4 cpu = 60%

primes = 4cpus = 400% stupid algorithm O(n sqrt(n))
mandelbrot = 4cpus = 400% O(w * h * iter)
matrix mult = 4cpus = 200% O(n^3)
sorting???? TOO MUCH MEMORY, too little CPU O(n log n)

Example of omp reduction

```c++
float sum = 0.0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < n; i++) {
    sum += a[i];  // Each thread computes partial sum, combined automatically
}
```

Example of omp parallel for reduction (threading only, no SIMD)

```c++
float sum = 0.0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < n; i++) {
    sum += a[i];  // Parallelized across threads, may auto-vectorize with -O3
}
```

Example of omp parallel for simd reduction (threading + explicit SIMD)

```c++
float sum = 0.0;
#pragma omp parallel for simd reduction(+:sum)
for (int i = 0; i < n; i++) {
    sum += a[i];  // Both threaded and vectorized
}
```

Example of omp barrier

```c++
#pragma omp parallel
{
    // Each thread does independent work
    int local_result = do_work(omp_get_thread_num());
    
    #pragma omp barrier  // Wait for all threads to finish
    
    // Now safe to use results from all threads
    #pragma omp single
    combine_results();
}
```

Example of omp task

```c++
#pragma omp parallel
{
    #pragma omp single
    {
        for (int i = 0; i < 10; i++) {
            #pragma omp task
            {
                process_item(i);  // Tasks executed by any available thread
            }
        }
    }
}
```

- OpenMP variables and functions

```c++
int id = omp_get_thread_num();
int n = omp_get_num_threads();
int mt = omp_get_max_threads();
omp_set_num_threads(n);
```

Why is there a maximum number of threads?
On cuda, the max = 256 * 1024 * 1024 * 32

Each thread needs:
PC (where to execute)
SP (stack pointer)
and a STACK. THe stack stores local variables in memory. Hopefully some are in registers for speed
But each stack needs memory. Lots of threads = lots of memory

## CUDA

CUDA runs on two computers

- Identify which computer code is running on

```cuda
void f() { // runs on CPU

}

__forceinline__ __device__ void h() {

}

__global__ void g(int* p, int n) { // kernel, runs multithreaded on GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
     h();
}

// device functions cannot be called directly. Only from kernels
__device__ int* setMemory(uint32_t n) {


}



void main() {
    const int n = 1'000'000;
    int* p = malloc(n * sizeof(uint32_t)); // memory is on CPU
    for (int i = 0; i < n; i++)
      p[i] = i;
    int* dev_p;
    cudaMalloc(&dev_p, n * sizeof(uint32_t))
    // NO: illegal setMemory();
    // NUMA - Non-uniform memory access (not covered)

    cudaMemcpy(dev_p, p, n * 1024*sizeof(uint32_t),
          cudaMemcpyHostToDevice);
    g<<<256, 256>>>(dev_p, n * 1024);  // 256 blocks, 256 threads/block = 65,536 threads
    // g<<<1024, 128>>>(dev_p, n * 1024);  // 1024 blocks, 128 threads/block = 131,072 threads
    // g<<<64, 512>>>(dev_p, n * 1024);    // 64 blocks, 512 threads/block = 32,768 threads
    // g<<<1024, 1024>>>(dev_p, n * 1024); // 1024 blocks, 1024 threads/block = 1,048,576 threads

// can we write:
//    g<<<256, 7>>>(dev_p, n * 1024);  // 256 blocks, 256 threads/block = 65,536 threads


}

```
