#include <iostream>
#include <cstdint>
using namespace std;
#include <immintrin.h>
// Bubble Sort


#if 0
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
#endif

extern "C" void sort8(__m256i a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h);

void sort_avx(uint32_t arr[], uint32_t n)
{
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr));
    __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + 8));
    __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + 16));
    __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + 24));
    __m256i e = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + 32));
    __m256i f = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + 40));
    __m256i g = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + 48));
    __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + 56));

    sort8(a, b, c, d, e, f, g, h);

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr),      a);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + 8),  b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + 16), c);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + 24), d);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + 32), e);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + 40), f);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + 48), g);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + 56), h);
}
void print(uint32_t a[], uint32_t n) {
    for (uint32_t i = 0; i < n; i+=8) {
        for (uint32_t j = 0; j < 8; j++)
          std::cout << a[i+j] << '\t';
        std::cout << '\n';
    }
}

int main() {
    uint32_t a[] = {5, 1, 10, 2, 11, 14, 3, 6,
                    6, 2, 9,  3, 8, 6, 4, 7,
                    16, 12, 10, 4, 11, 12, 4, 7,
                    4, 2, 9,  3, 8, 6, 3, 7,
                    11, 2, 10, 3, 11, 2, 4, 7,
                    6, 2, 9,  3, 8, 6, 4, 7,
                    16, 12, 10, 4, 11, 12, 4, 7,
                    4, 2, 9,  3, 8, 6, 3, 7};
    sort_avx(a, 64);
    print(a, 64);
}

