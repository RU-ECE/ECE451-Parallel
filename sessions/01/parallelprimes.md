# parallel Primes homework

given numbers from 2 to n
how many cpus?  k

assume we all have 2..16 cores

try 1, 2, 3, 4, 8, ... 16, 32

go up to 4x the number of cores on your system

n=1000
k = 2

2..500          501...1000
                SLOWER because the numbers are bigger
                not ideal!

minimal homework 100% n threads will not achieve
n x speedup because the last one takes longest

+100% bonus

have a pool of threads
n=10^9
k = 4
thread 1: 1..10
thread 2: 11..20
thread 3: 21..30
thread 4: 31..40
thread 1: 41..50
...

partition idea #1
each thread takes i, i + k i+2k...

2, 2+8 = 10, 2+16=18, ...
problem: this one is too easy,

7, 7+8=15, 15+8=23, ...

partition
thread 1: 1...1,000,000
thread 2: 1,000,001 .. 2,000,000
thread 3:
thread 4:
thread 1: 5,000,001.. 6,000,000

