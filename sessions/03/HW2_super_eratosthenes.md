# HW2: Super mega parallel Eratosthenes multithreaded

Using the code shown in class that computes eratosthenes using bits computing only the odd numbers in each 64-bit word, write a multithreaded solution.


* 100% credit, just the basic solution that first calculates eratosthenes
  sequentially up to $\sqrt{n}$ and then breaks the solution into k pieces
  on k threads
* +20% if you write each word in the bit vector only once for each number.
  In other words, when cancelling all multiples of 3, if you can write all the bits
  into a register, and clear 9, 15, 21, 27, 33, 39, 45, 51, 56, 62
  then only write the resulting word ONCE.
* +20% recursive `parallel_eratosthenes`, so:
```cpp
uint64_t parallel_eratosthenes(uint64_t a, uint64_t b) {
   parallel_eratosthenes(a, sqrt(b)); // first 
}
```

base case for 100% credit:

```
uint64_t paralleeratosthenes(uint64_t n) {
  eratosthenes(2, sqrt(n)); // if n=1 billion, single thread 2..sqrt(n)
  // split sqrt(n)..n into k threads
  return sum of all primes found
}