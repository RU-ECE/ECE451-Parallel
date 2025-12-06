# Review: Memory and Multithreading Performance

## 1. Simple Multithreaded Sum Examples

Assume:

- `a` is a global (or shared) array of length `n`.
- We create `k` threads that all run the same function.

### 1.1. Summing the Whole Array

```c++
void f(int* ans) {
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    *ans = sum;
}

// Example usage:
std::thread t1(f, &ans1);
std::thread t2(f, &ans2);
// ...
````

Here, each thread:

* Reads **all** elements of `a`.
* Does a full pass from `0` to `n - 1`.

This is **memory-bandwidth heavy**: all threads are competing to read the entire array.

### 1.2. Summing a Single Element Repeatedly

```c++
void f(int* ans) {
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[0];
    *ans = sum;
}

// Example usage:
std::thread t1(f, &ans1);
std::thread t2(f, &ans2);
// ...
```

Here, each thread:

* Reads the **same** element `a[0]` repeatedly.
* Once `a[0]` is in cache, most accesses are very fast.

This is much less stressful on memory bandwidth, since:

* Only one location is being accessed.
* It stays in the cache, so very few main-memory accesses are needed.

---

## 2. Hyperthreading and Effective CPUs

With **hyperthreading**:

* If you have `k` physical cores, the OS might report **`2k` logical CPUs**.
* If you run up to `2k` threads at 100% CPU, and some execution units are idle (e.g., waiting on memory), the second
  hardware thread on each core can:
	* Use **otherwise idle** execution units.
	* Potentially improve throughput.

The realistic total work is roughly:

* Between **`k` and `2k`** times the throughput of a single thread.
* You do **not** get a full `2×` speedup from hyperthreading alone, but you can get **more than 1×**.

---

## 3. Memory and Out-of-Order / Vectorization

Modern CPUs use:

* **Out-of-order execution** to hide latencies (especially memory).
* **Vector registers** (SSE/AVX/AVX-512) to perform operations on multiple data at once.

One way to reduce the penalty of “waiting for memory” is to:

* Load data into **vector registers** and perform **column-wise** computations in parallel.

Conceptually:

```text
a1 a2 a3 a4
b1 b2 b3 b4
c1 c2 c3 c4
d1 d2 d3 d4
```

A vector register might hold `[a1, b1, c1, d1]` (one column), and you operate on all four values at once.
This can increase throughput and help hide memory latency if data is laid out and accessed efficiently.

---

## 4. Thread State

Each **thread** needs its own **architectural state**:

* **PC** (Program Counter)
	* On x86-64: `rip`, indicating where the thread’s code is executing.
* **SP** (Stack Pointer)
	* On x86-64: `rsp`, indicating the top of the thread’s stack.
* **Registers**
	* General-purpose registers (`rax`, `rbx`, etc.), SIMD registers, flags, etc.

Example:

* **Thread 1**: its own `PC`, `SP`, registers; shares most memory pages with other threads, may allocate some private
  data.
* **Thread 2**: its own `PC`, `SP`, registers; shares the same address space (same virtual memory) but has independent
  state.

Threads:

* **Share memory** (same address space).
* Have **separate execution state** (so the CPU can schedule them independently).
