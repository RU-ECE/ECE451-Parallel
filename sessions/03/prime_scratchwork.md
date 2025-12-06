# Prime Examples and Parallel Eratosthenes

## Basic Sieve Example

Let $n = 30$, so $\sqrt{n} \approx 5$.

We start with numbers:

```text
2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
1 1 1 1 1 1 1 1 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
````

After crossing out multiples of 2:

```text
1 1 0 1 0 ...
```

After crossing out multiples of 3 (and so on), eventually:

```text
1 1 0 1 0 1 0 0 0  1  0  1  0  0  0  1  0  1  0  0  0  1  0  1  0  0  0  1
```

After handling all primes up to $\sqrt{30} \approx 5$, we’ve marked composite numbers, leaving primes.

---

## Large Example

For a larger problem, say:

* $n = 1{,}000{,}000{,}000$
* $\sqrt{n} \approx 33{,}000$

We only need primes up to $\sqrt{n}$ to cancel composites up to $n$.

---

## Understanding How to Parallelize Eratosthenes

1. **Calculate up to $\sqrt{n}$ (sequential)**
   Use a standard (single-threaded) sieve to find all primes $\leq \sqrt{n}$.

2. **Parallel sieve from $\sqrt{n}$ to $n$**

	* Allocate:

	  ```c++
	  bool* isprime = new bool[n + 1];
	  ```

	* First, calculate primes up to $\sqrt{n}$ (sequential).

	* Then, run a **multithreaded sieve** on the range $[\sqrt{n}, n]$.

   Example for $n = 10^9$:

	* Compute primes up to $\sqrt{n} \approx 33{,}000$.
	* Divide the range $[\sqrt{n}, n]$ into `num_threads` pieces.

   Let:

   ```text
   a = sqrt(n)
   b = n
   ```

   For 4 threads:

	* Thread 1: `[a,                     a + (b - a)/4]`
	* Thread 2: `[a + (b - a)/4 + 1,    a + (b - a)/2]`
	* Thread 3: `[a + (b - a)/2 + 1,    a + 3(b - a)/4]`
	* Thread 4: `[a + 3(b - a)/4 + 1,   b]`

Each thread uses the same small list of primes $\leq \sqrt{n}$ to mark composites in its subrange.

---

## Wheel Factorization

### Basic Idea

* Using 2: skip even numbers.
* Using 2 and 3 together: the product is $2 \cdot 3 = 6$.

Example pattern over 6 to 29:

```text
6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
0, 1, 0, 0,  0,  1,  0,  1,  0,  0,  0,  1,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1
```

Here, the pattern of possible primes (1s) repeats with period 6 when we filter by 2 and 3.

For a larger wheel:

* Using primes 2, 3, 5, 7:

  $$2 \cdot 3 \cdot 5 \cdot 7 = 210$$

* This gives a repeating pattern of possible primes modulo 210.

---

## Writing Code with Scratchwork (Bit-Based Sieve)

We use **bits** to represent odd numbers only:

* 64 bits in a `uint64_t`.
* 2, 3, 5, 7 give a wheel of size 210.
* For a smaller wheel (2, 3, 5): $2 \cdot 3 \cdot 5 = 30$.

Example offsets (for a wheel of 30):

```text
8 bits: k+1, k+7, k+11, k+13, k+17, k+19, k+23, k+29
```

Skeleton code:

```c++
for (uint64_t i = a; i <= b; i += 30) {
    if (isprime(i + 1)) {
        // handle i+1
    }
    if (isprime(i + 7)) {
        // handle i+7
    }
    if (isprime(i + 11)) {
        // ...
    }
    // etc.
}
```

---

## Bit Vector Implementation

We want operations:

* `clear(i)`   — mark number `i` as composite.
* `isPrime(i)` — test if `i` is still marked as prime.
* Possibly **bulk operations** (update a whole 64-bit word at once).

Initialization example:

```c++
uint64_t num_words = (n + 63) / 64;  // number of 64-bit words

// For odd-only storage, memory is roughly halved:
// n = 1,000,000,000 -> ~125 MB if storing all bits, ~62.5 MB if odd-only.

uint64_t* isprime = new uint64_t[num_words];
for (uint64_t i = 0; i < num_words; i++)
    isprime[i] = 0xFFFFFFFFFFFFFFFFULL;  // assume all bits = "prime" initially
```

We can precompute offsets inside one wheel period:

```c++
const uint32_t offsets[] = {1, 7, 11, /* ... */};

for (uint64_t i = a; i <= b; i += 30) {
    for (uint32_t k = 0; k < 8; k++) {
        if (isPrime(i + offsets[k])) {
            // ...
        }
    }
}
```

Simpler boolean version (non-bit):

```c++
bool* isprime = new bool[n + 1];

for (int i = 3; i <= n; i += 2)
    isprime[i] = true;
```

---

## Memory Notes

* RAM = **Random Access Memory** (not “random” in the colloquial sense, but addressed directly).
* **Sequential access is fastest**:

	* Typical DRAM can return 8 words in a **burst** on one bank → 8 reads.
	* With 2 banks: up to 16 words in quick succession.
* Example latencies:

	* Within a **row** (same row buffer): CAS (column access) ≈ 30 clock cycles.
	* Switching to another **row** in the same bank: ~35 clock cycles.
	* Switching to a different **page** (plus memory management overhead) is even slower.

Fastest prime sieves often use **segmented sieving** so the working set fits in cache (e.g. 256 KB).

---

## How to Implement a Bit Vector (Conceptual)

We need:

* `clear(i)`:

	* Compute word index and bit index, then `word &= ~mask`.
* `isPrime(i)`:

	* Compute word index and bit index, then test `word & mask`.

We can also precompute a **repeating pattern** for the wheel:

* First calculate the first 105 words (or 210 numbers) for the wheel $2 \cdot 3 \cdot 5 \cdot 7 = 210$.
* Note that:

	* $64 = 2^6 = 2 \cdot 2 \cdot 2 \cdot 2 \cdot 2 \cdot 2$
	* $3 \cdot 5 \cdot 7 = 105$
	* $\text{lcm}(64, 105) = 64 \cdot 105$ (since 64 and 105 are coprime)
* This defines a large repeating block that can be reused.

---

## Small Parallel Example ($n = 30$)

Single-threaded sieve up to 5:

* Remove multiples of 2, 3, 5 in [2..5].

Remaining range [6..30], divide into 4 threads:

* Thread 1: `6..12`
* Thread 2: `13..18`
* Thread 3: `19..24`
* Thread 4: `25..30`

Each thread:

* Uses the small set of primes (2, 3, 5) to mark composites in its subrange.

---

## DRAM Timing Mnemonics

* **RAS** (Row Address Strobe), **CAS** (Column Address Strobe).
* Example:

	* 35 cycles to open a row (RAS),
	* 30 cycles CAS within the row,
	* Switching rows or pages adds extra overhead.

These details matter for high-performance prime sieves that try to keep everything inside **cache** as much as possible.
