# Test 1 review

## Topics

- homeworks
  - dot product
  - brute force prime
  - eratosthenes
  - mandelbrot
     divergence

     count = 3 3 3 3 3 3 3
     inc   = 1 1 1 1 1 1 1
     count += inc;
             4 4 4 4 4 4 4
    to stop counting, zero that one in inc
     inc =   1 1 1 0 0 1 1
- threading
  - What does a thread need (context)
  - How much does context switching cost (thrashing)
  - synchronization is slow
  - How much faster can multithreading be?
    If threads are waiting, speedup is enormous. We typically deal with
    CPU intensive loads.
  

- multi-processing
  - everything a thread needs
  - Memory Management Unit MMU entries
- vectorization
  - How much faster are vector operations?
  - What is the limitation on vector performance?
  - Memory issues (see below)

- size of things
  - memory bus 64 bits
  - vector register avx = 256 bits
  - *optional: vector avx512 = 512 bits
  - float = 4byte = 32bits
  - double = 8bytes = 64 bytes
  - avx register can be viewed as:
    2 128-bit registers
    4 64-bit registers
    8 32-bit register
    16 16-bit registers
    32 8-bit registers
- Memory
  - Memory access is the problem
  - shared bandwidth on the bus
  - shared bandwidth to Memory
  - DDR5 rules
    - RAS, CAS, Precharge  (ex. 46-45-45)
    - burst mode = 16
    - we showed that going out of order is bad, but we don't fully understand
      why the numbers are the way they are
  - demonstration of how much faster sequential vector reads are than normal reads
    - memcpy
  - What is the memory bandwidth of my machine?
    - sequential
    - random access
- C++ vs. Assembler
  - when the optimizer runs, weird things happen
  - you need to know what is being generated even if you don't write 
    assembler on a regular basis
  - pass by reference is evil!
  - taking the address of things is evil! (causes the optimizer to give up!)
 
 ```cpp
 thread t1(f);
 thread t2(f, 5);
 t2. join();

 mutex m;
 m.lock();
 m.unlock();
 ```

 we make fast parallel code not by mutex
 by partitioning the data
 so we don't need a mutex

 it's ok to use mutexes occasionally
 do a lot of computation, keep it granular



 1. Given the following dot product code, how much faster can you make a vectorized using avx2?

```c++
 double dot(const float a[], const float b[], int n) {
    double sum;
    for (int i = 0; i < n; i++))
        sum += a[i] * b[i];
    return sum;
 }
 ```

 first cpu = 100% 2nd = 80% 3rd = 60% 4th = 40%


