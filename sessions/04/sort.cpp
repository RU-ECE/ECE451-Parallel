#include <immintrin.h>
// Bubble Sort



void sort(uint32_t a[], uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = i + 1; j < n; j++) {
            if (a[i] > a[j]) {
                uint32_t temp = a[i];
                a[i] = a[j];
                a[j] = temp;
            }
        }
    }
}

void sort_avx(uint32_t arr[], uint32_t n) {
	__mm256i a = _mm256_loadu_pd(arr); // load first 8 numbers 32x8 = 256
	__mm256i b = _mm256_loadu_pd(arr+8);
  __mm256i c = _mm256_loadu_pd(arr+16);
	__mm256i d = _mm256_loadu_pd(arr+24);
	__mm256i e = _mm256_loadu_pd(arr+32);
	__mm256i f = _mm256_loadu_pd(arr+40);
	__mm256i g = _mm256_loadu_pd(arr+48);
	__mm256i h = _mm256_loadu_pd(arr+56);

	



	
}



