# Homework 4: Threading and Memory Access

[Register](https://en.wikipedia.org/wiki/Processor_register) | [Cache](https://en.wikipedia.org/wiki/Cache_(computing))

---

## 1. Basic Facts

1. What does a computer's clock speed regulate?  
   The speed of ________________

2. We double the clock speed:
	1. memory read is __________
	2. memory write is _________
	3. reading from cache is _______
	4. multiplication speed is _______

3. Speed of each instruction in clock cycles:
	1. `add %rax, %rbx`  _________
	2. `mul %rax, %rbx`  _________
	3. `div %rcx`        _________
	4. `shl $1, %rax`    _________

4. Assume the following memory read instructions are in cache,  
   but the location being read is new. timing is 46–45–45:
	1. `mov    (%rsi), %rax`        _________
	2. `vmovdqa (%rsi), %ymm0`      __________

5. Why can't we just double the clock speed to make our computer faster?
   _______________

6. What is the fastest memory in a computer? ________________

7. What is the 2nd fastest memory? ________________

8. What makes main memory (DRAM) so much slower than the fastest memory?
   ________________

9. How many integer registers are there on an x86-64 machine? ________________

10. How many vector registers are there on an AVX2 machine? ________________

11. How many integer registers are there on an ARM64 machine? ________________

12. Why might the ARM designers have chosen differently than Intel?
    ________________

13. A special register called RIP on Intel: what does it stand for?
    _____________________

14. Look up what it does: _________?

15. What is the register RSP on Intel? _____________________

16. What is L1 cache on x86 architecture?  
    number of cores __________, size ____________

17. What is L2 cache on x86 architecture?  
    number of cores __________, size ____________

18. What is L3 cache on x86 architecture?  
    number of cores __________, size ____________

19. Approximately how long does it take light to travel 30 cm? _______________

---

## 2. Class Survey

1. Enter your name into a row of the spreadsheet and complete the data for your computer.
2. If you have more than one computer you can pick the one you use. You may enter more than one.

   <https://docs.google.com/spreadsheets/d/10DiQJcTMTqcE1JjSKFx0AWUcOAqu75cQUcKsF0jtxBg/edit?usp=sharing>

---

## 3. Basics of Multiprocessing

1. What is a process?

2. What is a thread?

3. Every thread requires at a minimum `sp`, `pc` / `rip`. Explain.

4. A computer with 4 cores is running a job.  
   1 thread, `t = 10s`  
   2 threads, `t = 5s`  
   4 threads, `t = 3s`

   Neglecting hyperthreading, why might it not be a good idea to run with 8 threads?

---

## 4. Explain the Benchmark Results for `memory_timings.s`

1. For each function run, explain (in one line) what it is attempting to measure.

2. Why is `write_one` so much slower than `read_one`?

3. Why is `read_memory_avx` faster than `read_memory_scalar` if both are reading the same amount of memory sequentially?

4. Run on your computer (or a lab computer if yours is an ARM or you have some other problem). Report the results.
	1. Extra credit: if you write your own assembly code and test a different architecture, **+50%**.
	2. Find the CPU and memory configuration for the machine you tested.
		- This can be **About...** in Windows, or
		- In Linux you can use `lscpu` and `cat /proc/cpuinfo`.

---

## 5. Explain What a Cache Is

1. Explain what a **cache miss** is.

2. Why is `read_memory_repeated1` faster than `read_memory_repeated2`?

---

## 6. Why is it so important to understand memory performance for parallel computing?

---

## 7. Why does pipelining on a modern CPU make benchmarking so difficult?

---

## 8. Why doesn't AMD define their own extensions to the instruction set, like more registers?

---

## 9. Why doesn't a CPU manufacturer design a computer with 1 million registers?
