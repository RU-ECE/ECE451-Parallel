# Test 1 Review

## Topics

- Homeworks
- Threading and multiprocessing
- Vectorization
- Memory performance
- C++ vs assembler / optimization
- Example questions (dot product, scaling)

---

## 1. Homeworks

Topics that may come up:

- **Dot product**
- **Brute-force prime checking**
- **Eratosthenes sieve**
- **Mandelbrot**
	- **Divergence** and counting iterations

Example of divergence tracking with a “mask” / increment vector:

```text
count = 3 3 3 3 3 3 3
inc   = 1 1 1 1 1 1 1

count += inc
count = 4 4 4 4 4 4 4
````

To **stop counting** for some elements (e.g., those that have already diverged), zero out those positions in `inc`:

```text
inc = 1 1 1 0 0 1 1
```

Now only the elements with `inc = 1` will keep increasing `count`.
This is essentially how branchless / SIMD Mandelbrot code avoids divergence problems by masking out “done” lanes.

---

## 2. Threading

Concepts:

* What does a **thread** need (its **context**)?
	* Program counter (PC / `rip`)
	* Stack pointer (`sp` / `rsp`)
	* Registers, etc.
* **Context switching cost** (thrashing):
	* Saving/restoring registers, stacks, etc. is not free.
	* Too many threads → overhead dominates.
* **Synchronization is slow**:
	* Locks, mutexes, condition variables, etc.
	* We try to **minimize** synchronization.
* How much faster can multithreading be?
	* If threads are mostly **waiting** (I/O bound), speedups can be enormous by overlapping waits.
	* In this course, we mostly deal with **CPU-intensive** loads, so speedup is bounded by:
		* Number of cores
		* Memory bandwidth
		* Parallel portion (Amdahl's Law)

Example CPU utilization pattern:

```text
1st CPU = 100%
2nd CPU = 80%
3rd CPU = 60%
4th CPU = 40%
```

This illustrates that as we add more threads, we **don’t necessarily get linear scaling**, especially once we saturate
some shared resource (often memory).

---

## 3. Multi-processing

* A **process** needs everything a thread needs **plus**:
	* Its own address space, page tables, MMU entries, etc.
* Context switching between processes is usually **heavier** than between threads.

---

## 4. Vectorization

Questions to think about:

* How much faster are **vector operations**?
	* AVX (256-bit) can operate on:
		* 4 doubles (64-bit)
		* 8 floats (32-bit)
		* 16 shorts (16-bit)
		* 32 bytes (8-bit)
* What limits vector performance?
	* **Memory bandwidth**
	* Unaligned or scattered data
	* Branching / divergent control flow
	* Data dependencies

### Size of Things

* Memory bus: typically **64 bits** wide
* Vector register (AVX): **256 bits**
* Optional: AVX-512: **512 bits**
* Types:
	* `float` = 4 bytes = 32 bits
	* `double` = 8 bytes = 64 bits

An AVX (256-bit) register can be viewed as:

* 2 × 128-bit chunks
* 4 × 64-bit lanes
* 8 × 32-bit lanes
* 16 × 16-bit lanes
* 32 × 8-bit lanes

---

## 5. Memory

Key idea: **Memory access is usually the problem**.

* **Shared bandwidth** on:
	* The memory bus.
	* Access to DRAM.
* DDR5 rules of thumb:
	* RAS, CAS, Precharge timings (e.g., `46-45-45`)
	* Burst mode = 16 (once you pay the initial cost, subsequent accesses are much faster)
	* We showed that **out-of-order / random access is bad**, but the detailed timing behavior is complex.
* Demonstration:
	* **Sequential vector reads** (e.g., via `memcpy` or AVX loads) are much faster than scattered scalar reads.

Questions you should be able to answer:

* What is the **memory bandwidth** of your machine?
	* Sequential access (best case)
	* Random access (worst case)

---

## 6. C++ vs Assembler

* When the **optimizer** runs, “weird things” can happen:
	* Dead code elimination
	* Loop unrolling
	* Vectorization
	* Reordering of operations
* You don’t have to write assembler regularly, but you should have a rough idea of:
	* What code the compiler **generates**
	* How your C++ patterns map to assembly

Cautions:

* **Pass by reference** can be tricky for optimization:
	* The compiler must assume references can alias.
* **Taking the address** of variables can also force the compiler to be conservative:
	* It may give up on some optimizations (like keeping things purely in registers).

---

## 7. Mutex vs Partitioning

Example threading snippet:

```c++
std::thread t1(f);
std::thread t2(f, 5);
t2.join();

std::mutex m;
m.lock();
m.unlock();
```

General rule:

* We make **fast parallel code** **not** by using lots of mutexes, but by:
	* **Partitioning data** so that each thread works on its own chunk.
	* Avoiding shared writable state wherever possible.
* It’s OK to use mutexes occasionally:
	* But keep the **critical section small**.
	* Do lots of computation **outside** the locked region.
	* Keep work **granular**.

---

## 8. Example Question: Dot Product Vectorization

> 1. Given the following dot product code, how much faster can you make it using AVX2?

```c++
double dot(const float a[], const float b[], int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}
```

Things to consider when answering:

* How many float multiplications/additions can fit into one AVX2 instruction?
* What is the theoretical speedup from vectorization **alone**?
* What other bottlenecks (memory bandwidth, alignment, etc.) might reduce the real-world speedup?
