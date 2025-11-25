#include <cstdint>
using namespace std;

/*
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
    1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1, 
    1  1  0     0     0      0       0       0       0       0       0       0       0       0       0       0       0
                0        0           0           0           0           0           0           0           0
                             0                   0                   0                   0   
*/

uint64_t eratosthenes(uint64_t n) {
    uint64_t count = 0;
	auto isPrime = new bool[n + 1];

    // first assume all numbers are prime
    for (uint_64t i = 2; i <= n; i++)
      isPrime[i] = true;

    for (uint64_t i = 2; i <= n; i++) {
        if (isPrime[i]) {
            count++;
            for (uint64_t j = 2 * i; j <= n; j += i)
                isPrime[j] = false;
        }
    }
    delete [] isPrime;
    return count;    
}

/*
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
    1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1, 
    1  1  0     0     0      0       0       0       0       0       0       0       0       0       0       0       0
                         0                       0                       0

    O(n log log n)
*/

uint64_t improved_eratosthenes(uint64_t n) {
    uint64_t count = 0;
	auto isPrime = new bool[n + 1];

    // first assume all numbers are prime
    for (uint_64t i = 2; i <= n; i++)
      isPrime[i] = true;

    for (uint64_t i = 2; i <= n; i++) {
        if (isPrime[i]) {
            count++;
            for (uint64_t j = i * i; j <= n; j += 2*i)
                isPrime[j] = false;
        }
    }
    delete [] isPrime;
    return count;    
}


bool isPrime(uint64_t* primes, const uint64_t i) {
    return primes[i/64] & (1ULL << (i & 63));
// 8421
// 1010
//  101
// 10101001010101010110X1011001010101010011111111111111111111
// 0000000000000000000000000000000000000000000000000000000001
// i % 64 == i & 63 
//              1000000
//                 111111 

}

/*
    suggested optimizations
    don't store even numbers

    use prime number wheel
    2, 3 = 2*3 = 6

    6,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23,
    0    1  2,  3,   4,  5
         X               X 
    mod(6)


     2,3, 5, 7 = 210
    3, 5, 7 = 105

    105 is not an even multiple of 64
    lcm(105, 64) = 105*64 = 6720

https://en.wikipedia.org/wiki/Wheel_factorization

    segmented eratosthenes


    1st pass is the worst: n/2
    2nd pass: n/3
    3rd pass: n/5
    4th pass: n/7
*/
uint64_t improved_bitpacked_eratosthenes(const uint64_t n) {
    uint64_t count = 0;
    const uint64_t SIZE = (n + 63+1) / 64;
	auto primes =  new uint64_t[SIZE];
    // first assume all numbers are prime
    for (uint_64t i = 2; i <= SIZE; i++)
      isPrime[i] = 0xFFFFFFFFFFFFFFFFLL; // MAX(uint64_tuint64_t

    for (uint64_t i = 2; i <= n; i++) {
        if (isPrime(primes, i)) {
            count++;
            for (uint64_t j = i * i; j <= n; j += 2*i)
                clearPrime(primes, j);
        }
    }
    delete [] isPrime;
    return count;    
}


/*
    Hard to parallelize first step, because we need to compute numbers up to sqrt(n)
    in order to do eratosthenes up to n

    Stanley: can we come up with some kind of algorithm to split the first kernel


*/