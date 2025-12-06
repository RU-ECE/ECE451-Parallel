# Summary of Parallel Programming Taught in This Course

## Course Outline

- **Overview of Parallel Programming**
	- What is Parallel Programming?
	- Mechanisms of parallelism we will cover:
		- **C++**
			- Threads
			- Bit operations
		- **SIMD (Vectorization)**
			- AVX intrinsics (C++ functions wrapping Intel SIMD instructions)
			- Optional: use NEON for ARM (for anyone with an ARM CPU)
		- **OpenMP**
			- Automated parallelization using threads and SIMD
			- Extra hints in pragmas tell the compiler how to generate parallel code
		- **GPUs**
			- CUDA
			- Optional: HIP (AMD open source GPU compute API)
		- **MPI (Message Passing Interface)**
			- Run multiple computers in parallel
			- Synergistic with OpenMP, GPU programming, etc.

- **Introduction to Parallel Computing: Threads**
	- C++ programming skills
		- Modern C++: C++11/14/17/20/23
		- Debugging with `gdb`
			- IDE debuggers: CLion, VS Code
			- Linux debugging tools: Valgrind, AddressSanitizer, MemorySanitizer, ThreadSanitizer
	- C++ threads
	- Mutexes
	- Limitations to parallel execution
		- Memory bandwidth and characteristics
		- Cache behavior
		- Branch prediction
		- Race conditions
		- Deadlock
	- Using bit operations to reduce memory bandwidth
		- Eratosthenes’ sieve example
		- Using special instructions like `popcnt` to parallelize operations
	- Writing high-performance C++

- **SIMD (Single Instruction, Multiple Data) Computing**
	- AVX intrinsics (C++ wrappers for Intel SIMD instructions)
	- Naming conventions for Intel intrinsics

- **OpenMP**
	- Parallelizing loops
	- Parallelizing regions
	- Parallelizing functions
	- Summation / reduction
	- Using OpenMP pragmas
	- Using the OpenMP API

- **CUDA**
	- Basics of GPU computing model
	- Kernels, grids, blocks, and threads
	- Host vs device memory
	- Simple performance considerations

---

## Memory Architecture

- **Memory speed is inversely related to size**
	- Smaller memory is faster.
		- **Fastest memory:** CPU **registers**
		- **CPU cache** (on x86, typically L1, L2, L3):
			- **L1 cache**: ~64–256 KB per core (very fast, private to a core)
			- **L2 cache**: ~256 KB–1 MB (often per core or small core cluster)
			- **L3 cache**: ~4–32 MB (usually shared by all cores)
		- **Main memory (DRAM)**: larger, much slower than caches
		- **Nonvolatile storage:**
			- SSD (solid state drive, slower than RAM)
			- Hard drive (slowest, mechanical)

- On a single computer, there is **one main memory system**, often with **2 memory banks** on a typical PC.

- **Memory bandwidth** = how much data can be transferred per second.
	- Highly dependent on the **pattern of usage**:
		- **Read the same memory location repeatedly** → stays in cache, very fast.
		- **Sequential access** (ideal for RAM, uses full rated bandwidth):
			- Usually enough for more than 1 core running full speed.
			- Often **not enough** to feed more than 2 cores at full speed unless your computation does a lot of work
			  per load.
			- DDR4/DDR5 RAM supports **burst mode**:
				- First access in a burst is slow.
				- Next ~7 accesses in the same burst are ~1 clock each.
			- With **2 banks**, you can interleave bursts → up to 16 sequential accesses in rapid succession.

		- **Timing details** (example RAM):
			- RAM timings (
			  from [Tom's Hardware](https://www.tomshardware.com/reviews/cas-latency-ram-cl-timings-glossary-definition,6011.html)):
				- CL timing fields:
					- **CAS Latency (CL)**
					- **RAS to CAS Delay (tRCD)**
					- **RAS Precharge Delay (tRP)**
				- Example: Micron 32 GB DDR5-5600 ECC UDIMM 2Rx8 timings: `46-45-45`
					- Access time within a row:
						- $46$ clocks $\times \frac{1}{5600\text{ GHz}} \approx 8.2 \text{ ns}$
					- Access time between rows:
						- $45$ clocks $\times \frac{1}{5600\text{ GHz}} \approx 8 \text{ ns}$ additional
					- Access time between banks:
						- Another $45$ clocks $\approx 8 \text{ ns}$ additional

			- About **2%** of DRAM bandwidth is used for **refresh**.

		- **Skipping (non-adjacent) memory**:
			- Breaks sequential access patterns.
			- Forces more row/bank changes and loses burst-mode advantages.
			- Effective bandwidth drops and latency increases.

- **Key points about memory bandwidth:**
	- Memory bandwidth is **limited** and is a **property of the memory system**.
	- It is usually measured in **GB/s**.
	- For many-core systems, memory bandwidth often becomes the **bottleneck** unless you:
		- Reuse data heavily from cache.
		- Do a lot of computation per byte loaded (high arithmetic intensity).
