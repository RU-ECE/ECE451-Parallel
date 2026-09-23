#include <iostream>
#include <cstdint>
#include <immintrin.h>

uint64_t f1(uint64_t a, uint64_t b) {
    return a + b;
}

uint64_t f2(uint64_t a, uint64_t b) {
    return a - b;
}

uint64_t f3(uint64_t a, uint64_t b) {
    return a * b;
}

uint64_t f4(uint64_t a, uint64_t b) {
    return a / b;
}

double f1(double a, double b) {
    return a + b;
}

double f2(double a, double b) {
    return a - b;
}

double f3(double a, double b) {
    return a * b;
}

double f4(double a, double b) {
    return a / b;
}

__m256d f5(__m256d a, __m256d b) {
    return _mm256_add_pd(a, b);
}

__m256 f6(__m256 a, __m256 b) {
    return _mm256_add_ps(a, b);
}

__m256 f7(__m256 a, __m256 b) {
    return _mm256_sub_ps(a, b);
}

/*
    So, how would we sort values? We would have to compare and swap
    This is a problem in vector registers:
    ymm0 = [5, 1, 2, 9, 3, 1, 4, 8]
    ymm1 = [3, 2, 1, 4, 5, 9, 8, 7]

    what you cannot do with vector registers is use an if statement

    if (ymm0 > ymm1) {
        swap(ymm0, ymm1)
    }

    AVX has a great way to say this:

    temp = ymm0
    ymm0 = min(ymm0,ymm1)
    ymm1 = max(temp, ymm1)

    [(0,2),(1,3),(4,6),(5,7)]
[(0,4),(1,5),(2,6),(3,7)]
[(0,1),(2,3),(4,5),(6,7)]
[(2,4),(3,5)]
[(1,4),(3,6)]
[(1,2),(3,4),(5,6)]

*/

