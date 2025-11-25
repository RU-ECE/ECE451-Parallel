# Homework 1

Compute the sum $\frac{1}{i}$ as a double from $i = 1$ to $n = 100000000$, such
that $\frac{1}{1} + \frac{1}{2} + \frac{1}{3} + \ldots + \frac{1}{n}$

Compute forwards, as above, and backwards ($\frac{1}{n} + \frac{1}{n-1} + \frac{1}{n-2} + \ldots + \frac{1}{1}$)

Backwards is more accurate than forward!
1.23<br/>
.0887<br/>
.0926<br/>
======

1.23<br/>
.0887<br/>
======<br/>
1.31<br/>
.0926<br/>
====<br/>
1.40

.0887<br/>
.0926<br/>
======<br/>
.181<br/>
1.23<br/>
1.41

1. benchmark it single threaded
2. benchmark with 2, 3, 4, 8 threads (we would expect that $n = 16$ would only get slower); the highest number of
   threads should be twice the number of cores

Put the benchmark numbers in the comments
