#include <iostream>
#include <cmath>
using namespace std;
/**
 *  A prime is a integer divisible only by itself and 1
 *  by def. 1 is not prime
 * 2, 3, 5, 7, 11, 13, ...
 */
uint64_t countprimes(uint64_t a, uint64_t b) {
    uint64_t count = 0;
    for (uint64_t i = a; i <= b; i++) {

        for (uint64_t j = 2; j < i; j++) // 1 + 2 + 3 + ... n = n(n+1)/2
          if (i % j == 0)
            goto NOT_PRIME;
        count++;
      NOT_PRIME: ;
    }
    return count;
}

//   28 = 1,2,4  ,7  14,  28
uint64_t countprimes2(uint64_t a, uint64_t b) {
    uint64_t count = 0;
    for (uint64_t i = a; i <= b; i++) {

        for (uint64_t j = 2; j <= sqrt(i); j++) // 1 + 2 + 3 + ... n = n(n+1)/2
          if (i % j == 0)
            goto NOT_PRIME;
        count++;
      NOT_PRIME: ;
    }
    return count;
}
/*
prime number wheel  2 * 3 * 5 = 30
partition with k =6
7, 8, 9, 10, 11, 12,     13, 14, 15, 16, 17, 18, ...

6n-1,  6n+1




*/

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    cout << countprimes2(2, n) << '\n';
    return 0;
}