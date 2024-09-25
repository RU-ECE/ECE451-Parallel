# HW4 Answer the following questions about threading and memory access

[Register](https://en.wikipedia.org/wiki/Processor_register)
[Cache](https://en.wikipedia.org/wiki/Cache_(computing))

1. Basic facts
    a. What is the significance of a computer's clock speed? ________________
    b. Do all instructions take the same number of clock cycles to execute? ________________
    c. Why can't we just double the clock speed to make our computer faster? _______________
    c. What is the fastest memory in a computer? ________________
    d. What is the 2nd fastest memory? ________________
    e. What makes main memory (DRAM) so much slower than the fastest memory? ________________
    f. How many integer registers are there on an x86-64 machine? ________________
    g. How many vector registers are there on an AVX2 machine? ________________
    h. How many integer registers are there on an ARM64 machine? ________________
    i. Why might the ARM designers have chosen differently than Intel? ________________
    j. A special called RIP on intel rip. What does it stand for: _____________________
    k. Look up what it does _________?
    l. What is the register RSP on intel? _____________________
    m. What is L1 cache on x86 architecture? number of cores __________, size ____________
    n. What is L2 cache on x86 architecture? number of cores __________, size ____________
    o. What is L3 cache on x86 architecture? number of cores __________, size ____________
    p. Approximately how long does it take light to travel 30cm? _______________

2. Class Survey
   a. Enter your name into a row of the spreadsheet and complete the data for your computer.
   b. If you have more than one computer you can pick the one you use. You may enter more than one.

2. Basics of Multiprocessing
    a. What is a process?
    b. What is a thread?
    c. Every thread requires at a minimum sp, rip (also called PC). Explain
    d. A computer with 4 cores is running a job.
       1 thread, t=10s, 2 threads t=5s, 4 threads t=3s
       Neglecting hyperthreading, Why might it not be a good idea to run with 8 threads? 

3. Explain the benchmark results for memory_timings.s.
   a. For each function run explain (in one line) what it is attempting to measure.
   b. Why is write_one so much slower than read_one?
   c. Why is read_memory_avx faster than read_memory_scalar if both are reading the same amount of memory sequentially?
   d. Run on your computer (or a lab computer if yours is an ARM or you have some other problem). Report the results.
     1. Extra credit: if you write your own assembly code and test a different architecture, + 50%
     2. Find the CPU and memory configuration for the machine you tested. This can be About... in windows, or in linux you can use lscpu and cat /proc/cpuinfo
5. Explain what a cache is
   a. Explain what a cache miss is
   b. Why is read_memory_repeated1 faster than read_memory_repeated2?

6. Why is it so important to understand memory performance for parallel computing?
7. Why does pipelining on a modern CPU make benchmarking so difficult?
8. Why doesn't AMD define their own extensions to the instruction set, like more registers?
9. Why doesn't a CPU manufacturer design a computer with 1 million registers?
