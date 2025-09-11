/*

hw1: COmpute the sum 1/i double
for i=1 to n=100'000'000
1/1 + 1/2 + 1/3 + ... + 1/n

compute forward and backward

1/n + 1/(n-1) + 1/(n-2) + ... + 1/1

1/100 + 1/99 + 1/98 + ... 1/1


backward is more accurate than forward!
1.23
 .0887
 .0926
======

1.23
 .0887
======
1.31
 .0926
====
1.40

 .0887
 .0926
======
  .181
 1.23
 1.41


1. benchmark it single threaded 
2. benchmark with 2,3,4,8 threads (we would expect that n=16 would only get slower)
highest number of threads should be 2x number of cores


put the benchmark numbers in the comments
*/