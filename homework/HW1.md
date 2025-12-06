# Homework 1

Compute the sum

$$
\sum_{i=1}^{n} \frac{1}{i}
$$

as a `double` for

- $n = 100000000$ (i.e., $10^8$),

so explicitly:

$$
\frac{1}{1} + \frac{1}{2} + \frac{1}{3} + \ldots + \frac{1}{n}.
$$

You will:

1. Compute the sum **forwards**, as above:
   $$
   \frac{1}{1} + \frac{1}{2} + \frac{1}{3} + \ldots + \frac{1}{n}
   $$
2. Compute the sum **backwards**:
   $$
   \frac{1}{n} + \frac{1}{n-1} + \frac{1}{n-2} + \ldots + \frac{1}{1}
   $$

Because of floating-point rounding, the **backwards** sum is more accurate than the forwards sum.

---

## Example of Rounding / Order Effects

Think of a toy example (not the real harmonic sum) where we add three numbers in different ways.

One order:

```text
1.23
 .0887
 .0926
======
1.40 (rounded)
````

Different grouping:

```text
1.23
 .0887
======
1.31   (rounded)
 .0926
====
1.40
```

Another grouping, adding the small numbers first:

```text
 .0887
 .0926
======
 .181   (rounded)
1.23
1.41   (rounded)
```

The **same three numbers** added in different orders can give **slightly different results** (1.40 vs 1.41), purely due
to floating-point rounding. Your large harmonic sum will show similar effects depending on the order of summation.

---

## Benchmarking

1. **Benchmark single-threaded** (1 thread):
	* Time how long it takes to compute the sum (forwards and/or backwards) with a single thread.

2. **Benchmark multi-threaded**:
	* Time your implementation with:
		* 2 threads
		* 3 threads
		* 4 threads
		* 8 threads
		* (Possibly higher, depending on your core count)
	* The **highest number of threads** you use should be **twice the number of physical cores** on your machine.
	* We would expect that using something like **16 threads** on a machine with fewer cores may eventually become *
	  *slower**, due to overhead and contention.

Record:

* The number of threads.
* The measured runtime (e.g., seconds or milliseconds).
* Any observed speedup or slowdown.

---

## Deliverable

* Implement the summation and benchmarking.
* **Put the benchmark numbers in comments at the top of your program**, for example:

```cpp
// Name: Your Name
// HW1: Harmonic sum benchmark
// Machine: 8-core CPU
// n = 100000000
// 1 thread:  X.XXX seconds
// 2 threads: Y.YYY seconds
// 4 threads: Z.ZZZ seconds
// 8 threads: ...
// 16 threads: ...
```
