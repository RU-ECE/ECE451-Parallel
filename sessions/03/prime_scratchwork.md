# Prime examples
n=30 sqrt(n) = 5
2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
1 1 1 1 1 1 1 1 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
1 1 0 1 0
1 1 0   0   0 0 0     0     0  0  0     0     0  0  0     0     0  0  0
1 1 0 1 0 1 0 0 0  1  0  1  0  0  0  1  0  1  0  0  0  1  0  1  0  0  0  1
1 1 0 1 0 1 0 0 0  1  0  1  0  0  0  1  0  1  0  0  0  1  0  0  0  0  0  1


n=1,000,000,000

sqrt(n) =~ 33,000

## Wheel factorization

2: skip even numbers
2,3: 2*3 = 6
6,7,8,9,10,11    12,13,14,15,16,17,   18,19,20,21,22,23, 24,25,26,27,28,29
0,1,0,0,0, 1

## WRiting code with scratchwork
Using bits to write fewer times

2, 3, 5, 7 = 2*3*5*7 = 210


64 = the number of bits in a uint64_t
2,3,5
8 bits  k+1, k+7, k+11, k+13, k+17, k+19, k+23, k+29


for (uint64_t i = a; i <= b; i+= 30) {
  if (isprime(i+1)) {
    ...
  }
  if (isprime(i+7)) {

  }
  if (isprime(i+11)) {
    
  }

}

//                                 32  16  8 4 2 1
// 11111111111111111111111111111111111111111111111
bool a[100];
a[0] = true;

uint64_t num_words = (n +63) / 64;// 1billion 1/8 = 125Mb odd only = 62.5MB
for (int i = 0; i < num_words; i++)
  isprime[i] = 0xFFFFFFFFFFFFFFFFL;

const uint32_t offsets[] = {1, 7, 11, ...};
for (uint64_t i = a; i <= b; i+= 30) {
  for (uint32_t k = 0; k < 8; k++) {
    if (isprime(i+k)) {
      ...
    }
  }

}

for (int i = 3; i <= n; i+=2)
  isprime[i] = true;


isprime[i]  isprimes[i+1] ...

RAM = Random ACcess Memory (not random)
sequential is fastest: 8 in burst, 2 banks 8*2 = 16
within a row CAS = 30 clock cycles 
another row = ~ 35 clock cycles
to a different page + 1M (memory manager delay also)

fastest systems for prime number (segmented) do it within cache size = 256k

## How to implement a bit vector
need clear(i)
isPrime(i)

bulk set

just calculate the first 105 words 210 2*3*5*7
64 = 2^6 = 2*2*2*2*2*2    3*5*7=105
lcm(64,105 ) = 64*105

## How fast can we initialize 2,3,5,7?

