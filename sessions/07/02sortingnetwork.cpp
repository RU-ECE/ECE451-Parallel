#include <iostream>
#include <immintrin.h>

// there is no good way that I know to make this work well in C++
// C++ pass by reference will force these values into memory
void sort8(__m256i& a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h) {
}

/*
https://bertdobbelaere.github.io/sorting_networks.html#N8L19D6
[(0,2),(1,3),(4,6),(5,7)]
[(0,4),(1,5),(2,6),(3,7)]
[(0,1),(2,3),(4,5),(6,7)]
[(2,4),(3,5)]
[(1,4),(3,6)]
[(1,2),(3,4),(5,6)]
*/
void sortingnetwork8(const uint32_t arr[], uint32_t n) {
    for (uint32_t i = 0; i < n; i += 64) {
        __m256i a = _mm256_loadu_si256((__m256i*)(arr+i));
        __m256i b = _mm256_loadu_si256((__m256i*)(arr+i+8));
        __m256i c = _mm256_loadu_si256((__m256i*)(arr+i+16));
        __m256i d = _mm256_loadu_si256((__m256i*)(arr+i+24));
        __m256i e = _mm256_loadu_si256((__m256i*)(arr+i+32));
        __m256i f = _mm256_loadu_si256((__m256i*)(arr+i+40));
        __m256i g = _mm256_loadu_si256((__m256i*)(arr+i+48));
        __m256i h = _mm256_loadu_si256((__m256i*)(arr+i+56));
        // a = [5, 1, 10, 2, 11, 14, 3, 6]
        // b = [6, 2, 9,  3, 8, 6, 4, 7]
        // temp[5, 1, 9, 2,  8, 6, 3, 6]
            __m256i temp = _mm256_min_epi32(a, c); // order all pairs in a,c
        c = _mm256_max_epi32(a, c);
        a = temp;

        __m256i temp = _mm256_min_epi32(b, d); // order all pairs in a,c
        d = _mm256_max_epi32(b, d);
        b = temp;

        _mm256_storeu_si256((__m256i*)(arr+i), a);
        _mm256_storeu_si256((__m256i*)(arr+i+8), b);
        _mm256_storeu_si256((__m256i*)(arr+i+16), c);
        _mm256_storeu_si256((__m256i*)(arr+i+24), d);
        _mm256_storeu_si256((__m256i*)(arr+i+32), e);
        _mm256_storeu_si256((__m256i*)(arr+i+40), f);
        _mm256_storeu_si256((__m256i*)(arr+i+48), g);
        _mm256_storeu_si256((__m256i*)(arr+i+56), h);   
        
    }
}