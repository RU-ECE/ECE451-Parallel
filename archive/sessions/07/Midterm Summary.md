# Midterm Summary

The midterm will focus on concepts and reasoning more than memorizing details. You may be asked to **explain behavior**
or **analyze short C++ code snippets**.

---

## 1. Memory and Performance

Be able to explain how **memory attributes affect a computation**, including:

- **Sequential access**
	- Why accessing memory in order is usually faster (cache lines, prefetching).

- **Multiple banks**
	- How using multiple memory banks can increase effective bandwidth.
	- What happens if all accesses hit the same bank vs. different banks.

- **Skipping large amounts (strided / sparse access)**
	- Why jumping around in memory (poor locality) leads to worse performance.

- **Cache hierarchy (L1, L2, L3)**
	- Differences between:
		- Level 1: very small, very fast.
		- Level 2: bigger, slower.
		- Level 3: largest on-chip cache, slowest of the three.
	- How cache hits/misses at different levels affect execution time.

---

## 2. Threading and Performance

Be able to explain how **threading** impacts performance:

- When threading **scales well** vs. when it **scales poorly**:
	- E.g., limited by compute vs. limited by memory bandwidth or contention.

- Why **increasing the number of threads without bounds does not help**:
	- Oversubscription
	- Contention for shared resources (memory, cache, etc.)
	- Diminishing returns and overheads.

- **Hyperthreading (SMT)**
	- What it is (multiple hardware threads per core).
	- How it can help hide latency.
	- When it helps, and when it might not.

---

## 3. Vectorization

Know the basics of **vectorization (SIMD)**:

- **When can vectorization help?**
	- Same operation on many independent data elements.
	- Regular memory access patterns (e.g., arrays, contiguous data).
	- Few branches or uniform control flow.

- When vectorization is **hard** or **not helpful**:
	- Heavy branching.
	- Data dependencies between elements.
	- Irregular memory access.

---

## 4. Modern CPU Features Affecting Computation

Understand how the following affect performance:

1. **Pipelining**
	- Multiple instructions in different stages at once.
	- How branches (`if` statements) can cause pipeline stalls or mispredictions.

2. **Cache misses**
	- Cost of different kinds of misses.
	- How poor locality (or large working sets) slow computations.

---

## 5. Question Styles

Questions may involve:

- **Given code (in C++)**:
	- Explain performance behavior.
	- Identify memory, threading, or vectorization issues.
	- Predict what will happen if you change the number of threads, data layout, etc.

**or**

- **You write small code snippets** that:
	- Would / would not be good candidates for **parallelization**.
	- Would / would not be good candidates for **vectorization**.
