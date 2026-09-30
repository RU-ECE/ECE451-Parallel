/*

HW02: Prime numbers using Eratosthenes mega uber bit tricks

1. to calculate primes up to (n)...
2. first calculate primes up to s = sqrt(n)
   a. recursively... calculate primes up to sqrt(s) EXTRA CREDIT

3. now do the rest in parallel!

2 ...   sqrt(N)  ....    N
  10 101010101
  use the little eratosthenes shared to wipe out the big numbers

  a = sqrt(n) + 1;
  size = (n - a) / 4


  thread t1(parallel_eratosthenes, a, a+size, &count1);
  thread t2(parallel_eratosthenes, a+size+1, a+2*size, &count2);
    thread t3(parallel_eratosthenes, a+2*size+1, a+3*size, &count3);
    thread t4(parallel_eratosthenes, a+2*size+1, a+3*size, &count3);
   join all threads
   add up all numbers

   DON'T FORGET THE NUMBERS FROM 2 to sqrt(n)!!! they count !!!
*/