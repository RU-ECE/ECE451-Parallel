# Test 2 Review

## Topics

- OpenMP
- CUDA

---

## OpenMP

### 1. Programming Model

- OpenMP is a **shared-memory parallel programming model**.
- You **add directives** (mostly `#pragma` lines) to existing C/C++/Fortran code.
- If the compiler does not understand a pragma, it **ignores** it and the program still runs single-threaded.

Example of an ignored pragma:

```c++
#pragma yak 5   // ignored if your compiler doesn't know "yak"
````

OpenMP is enabled with flags such as `-fopenmp` (GCC/Clang) or `/openmp` (MSVC).

---

### 2. Core OpenMP Pragmas

Common pragmas (and what they do conceptually):

```c++
#pragma omp parallel
#pragma omp critical 
#pragma omp parallel for
#pragma omp parallel for private(var)
#pragma omp parallel for shared(var)
#pragma omp parallel for firstprivate(var)
#pragma omp parallel for reduction(op:var)
#pragma omp simd
#pragma omp sections
#pragma omp section
#pragma omp single
#pragma omp master
#pragma omp atomic
#pragma omp parallel sections
#pragma omp barrier   // NOT ON TEST
#pragma omp task      // NOT ON TEST
```

* `#pragma omp parallel`

  Create a **parallel region**. Multiple threads execute the enclosed block.

* `#pragma omp critical`

  Only **one thread at a time** executes the enclosed block (mutual exclusion).

* `#pragma omp parallel for`

  Create a parallel region and **split loop iterations** across threads.

* `#pragma omp parallel for private(var)`

  Each thread gets its **own copy** of `var` (uninitialized inside the region).

* `#pragma omp parallel for shared(var)`

  All threads share `var`. Writes must be synchronized to avoid race conditions.

* `#pragma omp parallel for firstprivate(var)`

  Each thread gets a private copy of `var` **initialized** from the pre-parallel value.

* `#pragma omp simd`

  Ask the compiler to **vectorize** the loop (SIMD).

* `#pragma omp sections` / `#pragma omp section`

  Divide different blocks of code (sections) among threads.

* `#pragma omp single`

  Only **one** thread executes the block (not necessarily thread 0).

* `#pragma omp master`

  Only the **master thread** (thread 0) executes the block.

* `#pragma omp atomic`

  Make a single memory update **atomic** (no race) for simple operations.

* `#pragma omp parallel sections`

  Parallel region where each section runs on some thread.

* `#pragma omp barrier` (not on test)

  All threads wait until **every** thread reaches this point.

* `#pragma omp task` (not on test)

  Create a task that can be executed by any available thread.

---

### 3. Example: Shared Arrays

Arrays are **shared by default**, and each thread works on different indices:

```c++
int a[1000], b[1000], c[1000];

#pragma omp parallel for shared(a, b, c)
for (int i = 0; i < 1000; i++) {
    c[i] = a[i] * b[i];  // different i per thread, no conflict
}
```

---

### 4. `omp parallel for` and `omp simd` Together

Key questions:

1. **Where is the best place to put `#pragma omp parallel for`?**

	* On an **outer loop** where each iteration is relatively heavy and independent.

2. **Where is the best place to put `#pragma omp simd`?**

	* On an **inner loop** where each iteration does similar work on contiguous data (good for vectorization).

3. **Why is the combination so good? Aren’t we maxing out RAM?**

	* `omp parallel for` uses **multiple cores** (thread parallelism).
	* `omp simd` uses **vector instructions** inside each core (data parallelism).
	* Together you get **thread × SIMD** speedup.
	* Memory is still a limit, but:

		* For compute-heavy loops (lots of arithmetic per load), you can benefit from both.
		* Good locality and cache usage help avoid being completely memory-bound.

Example: matrix multiplication:

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

---

### 5. Reductions

A reduction collects results from all threads into a **single value**.

Example (sum of array):

```c++
float sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < n; i++) {
    sum += a[i];  // each thread computes a partial sum
}
// sum now contains the total
```

Threading + SIMD together:

```c++
float sum = 0.0f;
#pragma omp parallel for simd reduction(+:sum)
for (int i = 0; i < n; i++) {
    sum += a[i];  // threaded AND vectorized
}
```

---

### 6. Barrier and Task (for completeness)

Not on the test, but good to know:

```c++
#pragma omp parallel
{
    int local_result = do_work(omp_get_thread_num());

    #pragma omp barrier  // wait for all threads

    #pragma omp single
    combine_results();
}
```

Tasks:

```c++
#pragma omp parallel
{
    #pragma omp single
    {
        for (int i = 0; i < 10; i++) {
            #pragma omp task
            process_item(i);  // may run on any thread
        }
    }
}
```

---

### 7. OpenMP Helper Functions

Common functions:

```c++
int id  = omp_get_thread_num();   // thread ID
int n   = omp_get_num_threads();  // number of threads in the team
int mt  = omp_get_max_threads();  // maximum usable threads

omp_set_num_threads(n);           // ask OpenMP to use n threads
```

---

### 8. Amdahl’s Law & Scaling

The speedup from `n` CPUs depends on how much of the program is:

* **CPU-bound** (compute) vs
* **Memory-bound** or **sequential**.

This is captured by **Amdahl’s law**: the parallel speedup is limited by the **serial fraction** of the code.

Rough intuition examples:

* **Primes** (naive prime checking: $O(n \sqrt{n})$)

	* Very compute-heavy; all cores can stay busy.
	* 4 CPUs can (in theory) get close to **$4 \times$** speedup if memory is not the bottleneck.

* **Mandelbrot** (pixel-wise; $O(w \cdot h \cdot \text{iter})$)

	* Also very compute-heavy and embarrassingly parallel.
	* 4 CPUs can again approach **$4 \times$** speedup.

* **Matrix multiplication** ($O(n^3)$)

	* High arithmetic intensity (lots of math per load).
	* Good scaling, though caches and memory layout matter.

* **Sorting** ($O(n \log n)$)

	* Often more **memory-bound** due to many random reads/writes.
	* Adding more threads may hit a memory bandwidth wall.

Key takeaway: **more threads ≠ always faster**—it depends on the ratio of compute to memory and the fraction that can be
parallelized.

---

### 9. Why Is There a Maximum Number of Threads?

Each thread needs:

* A **Program Counter (PC)**: where the thread is executing.
* A **Stack Pointer (SP)**: where its stack lives.
* A **stack** in memory: for local variables, function calls, etc.
* Its own set of **registers** (architectural state).

Lots of threads = **lots of memory** for stacks and management overhead.
That’s why there is a **maximum** number of threads the system and runtime can handle efficiently.

On GPUs, the hardware can support a **huge** number of concurrent threads (for example, many thousands or more) but each
has limited state.

---

## CUDA

### 1. Two “Computers”: CPU and GPU

CUDA programs effectively run on **two different processors**:

* **Host (CPU)**:

	* Regular C/C++ code.
* **Device (GPU)**:

	* CUDA kernels and device functions.

You must keep track of **which code runs where** and where the **memory lives** (host vs device).

---

### 2. Function Annotations

* Normal C/C++ function (runs on CPU):

  ```cuda
  void f() {  // runs on CPU
  }
  ```

* Device-only helper function (called from GPU code):

  ```cuda
  __forceinline__ __device__ void h() {
      // runs on GPU, can be inlined into kernels
  }
  ```

* Kernel (entry point for GPU execution):

  ```cuda
  __global__ void g(int* p, int n) {  // runs multithreaded on GPU
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      h(); // device function
      // work using idx and p
  }
  ```

* Device function that cannot be called from host:

  ```cuda
  __device__ int* setMemory(unsigned int n) {
      // device-side allocation or setup
  }
  ```

Device functions (`__device__`) **cannot** be called from host code; only from kernels or other device functions.

---

### 3. Host Code Example

```cuda
void main() {
    constexpr auto n = 1'000'000;

    // CPU (host) allocation
    auto* p = static_cast<int*>(malloc(n * sizeof(unsigned int)));
    for (auto i = 0; i < n; i++)
        p[i] = i;

    // GPU (device) allocation
    int* dev_p;
    cudaMalloc(&dev_p, n * sizeof(unsigned int));

    // Copy data from CPU to GPU
    cudaMemcpy(dev_p, p, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Launch kernel: <<<numBlocks, threadsPerBlock>>>
    g<<<256, 256>>>(dev_p, n);  // 256 blocks * 256 threads/block = 65,536 threads

    // Other possible launch configurations:
    // g<<<1024, 128>>>(dev_p, n);  // 1024 * 128 = 131,072 threads
    // g<<<64, 512>>>(dev_p, n);    // 64 * 512  = 32,768  threads
    // g<<<1024, 1024>>>(dev_p, n); // 1024 * 1024 = 1,048,576 threads

    // Note: Each GPU has its own limits on blocks, threads per block, etc.
}
```

The index calculation inside the kernel:

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

gives each thread a unique ID (within 1D grid of 1D blocks).

---

### 4. Why So Many Threads on CUDA?

A GPU can support a **huge** number of threads (conceptually):

* Many blocks × many threads per block.
* Internally scheduled in **warps** or **wavefronts** (for example, 32 threads per warp).

Reason:

* Threads are **very lightweight** (small register file per thread, no large private stacks like CPU threads).
* When some threads are **waiting on memory**, others can be run to keep the ALUs busy.
* This is how GPUs **hide latency** and achieve high throughput.

Contrast with CPU threads:

* Each CPU thread typically has a **large stack** and heavier OS/thread overhead.
* You cannot reasonably run millions of OS threads like you can with GPU threads.
