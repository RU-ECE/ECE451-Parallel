#include <immintrin.h>

double sum(uint64_t a, uint64_t b, int n) {
    __m256d av = _mm256_set_pd(a, a+1, a+2, a+3);
    __m256d one = _mm256_set_pd(1, 1, 1, 1);
    __m256d sum = _mm256_set_pd(0, 0, 0, 0);
    __m256d four = _mm256_set_pd(4, 4, 4, 4);
    for (int i = 0; i < n; i+= 4) {
        __m256d inv = _mm256_div_pd(one, av);
        sum = _mm256_add_pd(sum, inv);
        av = _mm256_add_pd(av, four);
    }
    double result[4];
    _mm256_storeu_pd(result, sum);
    double total = 0;
    for (int i = 0; i < 4; i++) {
        total += result[i];
    }

} 