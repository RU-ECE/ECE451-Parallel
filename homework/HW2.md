# Homework 2: Super Mega Parallel Eratosthenes (Multithreaded)

Using the code shown in class that implements the **Sieve of Eratosthenes** using **bits** and **only odd numbers** in
each 64-bit word, write a **multithreaded** solution.

We will work with a sieve up to some maximum value `n` (for example, up to $10^9$). Each bit corresponds to an odd
number, and even numbers are skipped.

---

## 1. Basic Version (100% Credit)

Write a multithreaded program that:

1. **Sequential pre-sieve up to $\sqrt{n}$**
	- First, compute the sieve **sequentially** for all numbers up to $\sqrt{n}$.
	- This gives you all primes up to $\sqrt{n}$.
2. **Parallel marking from $\sqrt{n}$ to $n$**
	- Use the primes up to $\sqrt{n}$ to mark off composites in the range $\sqrt{n} \dots n$.
	- Break this upper range into **$k$ pieces** and process them using **$k$ threads**.
	- Each thread marks multiples of the primes in its subrange.
	- At the end, you can count (or sum) the primes.

**Base-case function (for 100% version):**

```c++
uint64_t paralleeratosthenes(uint64_t n) {
    eratosthenes(2, sqrt(n));   // if n = 1 billion, single thread handles 2..sqrt(n)
    // split sqrt(n)..n into k threads
    // return sum of all primes found
}
````

You may choose what to return (for example, number of primes or sum of primes), but be consistent and document it.

---

## 2. Bit-Efficient Writes (+20% Bonus)

Your sieve is stored as a **bit vector** in 64-bit words, using **only odd numbers**. For extra credit:

> **+20%** if you write **each 64-bit word at most once per prime** (per cancellation step).

That is, when cancelling multiples of a prime like `3`, instead of:

* Clearing bit for `9`,
* Then writing the word,
* Clearing bit for `15`,
* Writing again,
* Clearing `21`,
* Writing again, etc.,

you should:

1. Load the 64-bit word into a register.
2. Clear **all relevant bits in that word**:
	* For example, for 3: clear bits corresponding to 9, 15, 21, 27, 33, 39, 45, 51, 57, 63 (depending on the word).
3. **Write the updated word once**.

In other words, for each prime and each word, do:

* One load,
* Several bit operations in registers,
* One store.

This minimizes memory writes and makes the sieve more efficient.

---

## 3. Recursive `parallel_eratosthenes` (+20% Bonus)

For additional bonus credit:

> **+20%** for implementing a **recursive** `parallel_eratosthenes` structure.

Example sketch:

```c++
uint64_t parallel_eratosthenes(uint64_t a, uint64_t b) {
    parallel_eratosthenes(a, sqrt(b));  // first handle lower range recursively
    // then handle the remainder, possibly in parallel
}
```

Idea:

* Recursively solve the problem on the smaller range $[a, \sqrt{b}]$.
* Use that result (primes up to $\sqrt{b}$) to sieve the rest of $[a, b]` in parallel.
* This can give you a cleaner structure and possibly better reuse of logic.

You can integrate this idea with the base-case function above. For example, a top-level call might eventually do:

```c++
uint64_t paralleeratosthenes(uint64_t n) {
    eratosthenes(2, sqrt(n));   // base: sequential sieve up to sqrt(n)
    // then recursively or iteratively split sqrt(n)..n among k threads
    // and return the sum (or count) of primes
}
```

---

## Summary of Credit

* **100%**:
	* Sequential sieve up to $\sqrt{n}$,
	* Parallel marking from $\sqrt{n}$ to $n$ using $k$ threads.
* **+20%**:
	* Optimize bit operations so that for each prime and each 64-bit word, you **write that word only once**.
* **+20%**:
	* Implement a recursive `parallel_eratosthenes` structure as described.

Document:

* How many threads you used.
* Any speedups you observed compared to the single-threaded version.
