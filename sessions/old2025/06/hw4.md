# HW4 Answer the following questions about threading and memory access

[Register](https://en.wikipedia.org/wiki/Processor_register)
[Cache](https://en.wikipedia.org/wiki/Cache_(computing))

1. Basic facts
    a. What does a computer's clock speed regulate? The speed of ________________
    b. We double the clock speed
      1. memory read is __________
      2. memory write is _________
      3. reading from cache is _______
      4. multiplication speed is _______
    c. Speed of each instruction in clock cycles.
      1. add %rax, %rbx _________
      2. mul %rax, %rbx _________
      3. div %rcx       _________
      4. shl $1, %rax   _________
    d. Assume the following memory read instructions are in cache,
       but the location being read is new. timing is 46-45-45
      1. mov (%rsi), %rax       _________
      2. vmovdqa (%rsi), %ymm0 __________
    e. Why can't we just double the clock speed to make our computer faster? _______________
    f. What is the fastest memory in a computer? ________________
    g. What is the 2nd fastest memory? ________________
    h. What makes main memory (DRAM) so much slower than the fastest memory? ________________
    i. How many integer registers are there on an x86-64 machine? ________________
    j. How many vector registers are there on an AVX2 machine? ________________
    k. How many integer registers are there on an ARM64 machine? ________________
    l. Why might the ARM designers have chosen differently than Intel? ________________
    m. A special called RIP on intel rip. What does it stand for: _____________________
    n. Look up what it does _________?
    o. What is the register RSP on intel? _____________________
    p. What is L1 cache on x86 architecture? number of cores __________, size ____________
    q. What is L2 cache on x86 architecture? number of cores __________, size ____________
    r. What is L3 cache on x86 architecture? number of cores __________, size ____________
    s. Approximately how long does it take light to travel 30cm? _______________

2. Class Survey
   a. Enter your name into a row of the spreadsheet and complete the data for your computer.
   b. If you have more than one computer you can pick the one you use. You may enter more than one.
https://docs.google.com/spreadsheets/d/10DiQJcTMTqcE1JjSKFx0AWUcOAqu75cQUcKsF0jtxBg/edit?usp=sharing

2. Basics of Multiprocessing
    a. What is a process?
    b. What is a thread?
    c. Every thread requires at a minimum sp, pc/rip. Explain
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
