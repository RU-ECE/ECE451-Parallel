# Review Memory and Multithreading Performance

Create k threads

```cpp
void f(int* ans) {
int sum = 0;
for (int i = 0; i < n; i++)
  sum += a[i];
}
*ans = sum;

thread t1(f);
thread t2(f);
thread ...
```


```cpp
void f(int* ans) {
int sum = 0;
for (int i = 0; i < n; i++)
  sum += a[0];
}
*ans = sum;

thread t1(f);
thread t2(f);
thread ...
```

with hyperthreading k CPUs it looks you have 2k
if you have 2k threads running 100% CPU and some of your executaion units are not busy
   you MIGHT get more work done k <= work <2k


memory management

one way to solve the outof order penalty
use vector registers to do the column calculations in parallel

a1  a2  a3 a4
b1  b2  b3 b4
c1
d1


thread state

t1--> PC, SP, registers, shares all the memory manager pages, allocates some of its own



t2--> PC, SP, registers, shares all the memory manager pages










