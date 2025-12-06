# Homework: Parallel Primes

In this assignment you will parallelize a **prime-counting** program:

- Given integers from **2 to $n$**
- Using **$k$ worker threads or processes**
- Measure and analyze parallel speedup.

Assume your machine has between **2 and 16 cores**.

---

## 1. Basic Task (100%)

### 1.1. Problem

Write a program that:

- Counts (or lists) **prime numbers** in the range **$\left[2, n\right]$**
- Uses **$k$ parallel workers** (threads or processes)
- Measures execution time for:
	- **Sequential** version
	- **Parallel** versions with different values of **$k$**

You should experiment with **$k$** values:

- `1, 2, 3, 4, 8, ... 16, 32`
- In general, go up to **4$\times$ the number of physical cores** on your system.

Record:

- Your CPU model and number of cores.
- Timing results for each value of **$k$**.
- Observed speedup vs. the sequential version.

### 1.2. Simple Static Partitioning

Start with a **naive partitioning** strategy:

> Divide the interval $\left[2, n\right]$ into **$k$ contiguous chunks**, one per thread.

Example:

- Let **$n$ = 1000** and **$k$ = 2**.
- Then:
	- Thread 1: `2 .. 500`
	- Thread 2: `501 .. 1000`

**Issue:** This is **not ideal**.

- The second half (`501..1000`) contains **larger numbers** which are **more expensive** to test for primality.
- So the thread handling the larger numbers tends to run **longer**, causing **load imbalance**.
- This means even if you have $k$ threads, you will **not** achieve **$k$× speedup**, because the **slowest thread**
  determines when the entire job finishes.

You should:

- Implement this simple scheme.
- Measure how performance scales with increasing **$k$**.
- Comment on why you **do not** see ideal linear speedup.

---

## 2. Bonus: Dynamic Work Assignment (+100%)

For full bonus credit, implement a **better load-balancing strategy** using a **pool of threads** and a **shared work
queue** (or equivalent mechanism).

### 2.1. Setup

Consider a large problem size, e.g.:

- **$n$ = 10⁹**
- **$k$ = 4** worker threads

Instead of pre-assigning one big block per thread:

#### Dynamic scheme (example with chunk size 10):

- Thread 1: `1 .. 10`
- Thread 2: `11 .. 20`
- Thread 3: `21 .. 30`
- Thread 4: `31 .. 40`
- When a thread finishes, it grabs the **next available chunk**:
	- Thread 1: `41 .. 50`
	- Thread 2: `51 .. 60`
	- etc.

This approach:

- Automatically balances the work.
- Keeps all threads **busy** until the entire range is processed.

You may choose any reasonable **chunk size** (e.g., 100, 1000, …).

---

## 3. Partitioning Strategies (for Discussion / Comparison)

You may compare or explore different partitioning strategies:

### 3.1. Cyclic Partitioning

Each thread takes numbers of the form:

- Thread **i**: `start = i`, then `i + k`, `i + 2k`, `i + 3k`, …

Example (with `k = 8`):

- Thread handling `2`: `2, 10, 18, 26, ...`
- Thread handling `7`: `7, 15, 23, 31, ...`

This is a **cyclic** assignment. It may help distribute “hard” numbers more evenly, but can still have imbalance
depending on the cost of primality tests.

### 3.2. Large Contiguous Blocks

Another static idea:

- Thread 1: `1 .. 1,000,000`
- Thread 2: `1,000,001 .. 2,000,000`
- Thread 3: `2,000,001 .. 3,000,000`
- Thread 4: `3,000,001 .. 4,000,000`
- When all done, possibly:
	- Thread 1: `5,000,001 .. 6,000,000`
	- etc.

This can still suffer from **load imbalance** if some ranges are systematically more expensive than others.

---

## 4. What to Turn In

- Your **source code**:
	- Sequential version.
	- Parallel version(s) with clear indication of partitioning strategy.
- A **short writeup** (or comments at the top of your code) including:
	- Machine and CPU description.
	- Number of cores.
	- Values of **$n$** and **$k$** tested.
	- Timing results and observed speedups.
	- Brief discussion of:
		- Why the naive static scheme does **not** produce ideal speedup.
		- How the dynamic thread pool improves performance (if you implemented the bonus).
