#include <iostream>
#include <chrono>
#include <immintrin.h>
using namespace std;


void benchmark(string name, uint32_t (*f)(uint32_t[], int), uint32_t a[], int n) {
    auto t0 = chrono::high_resolution_clock().now();
    f(a, n);
    auto t1 = chrono::high_resolution_clock().now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    cout << name << " elapsed=" << elapsed << "ms\n";
}

uint32_t scan_array(uint32_t a[], int n) {
    for (int i = 0; i < n; i++)
        a[i];
    return 0;
}

uint32_t sum_array(uint32_t a[], int n) {
    uint32_t sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    return 0;
}

//                   %rdi          %rsi
uint32_t sum_array2(uint32_t a[], int n) {
    uint32_t sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    return sum;
}
/*
0:   0x0000000000000000
8:   0x0000000000000000
16:  0x0000000000000000
24:  0x0000000000000000
32:  0x0000000000000000
40:  0x0000000000000000
48:  0x0000000000000000
56:  0x0000000000000000
64:  0x0000000000000000


*/

/*
    AVX2 registers

    ymm0
    ymm1
    ymm2
    ymm3
    ymm4
    ymm5
    ymm6
    ymm7
    ...
    ymm15
*/
uint32_t hsum_epi32_avx(__m128i x)
{
    __m128i hi64  = _mm_unpackhi_epi64(x, x);           // 3-operand non-destructive AVX lets us save a byte without needing a movdqa
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));    // Swap the low two elements
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);       // movd
}

uint32_t hsum_8x32(__m256i v)
{
    __m128i sum128 = _mm_add_epi32( 
                 _mm256_castsi256_si128(v),
                 _mm256_extracti128_si256(v, 1)); // silly GCC uses a longer AXV512VL instruction if AVX512 is enabled :/
    return hsum_epi32_avx(sum128);
}    

uint32_t sum_array3(__m256i a[], int n) {
    __m256i sum = _mm256_setzero_si256(); // ymm0
    for (int i = 0; i < n; i += 8) {
        __m256i x = _mm256_loadu_si256(&a[i]); //ymm1?
        sum = _mm256_add_epi32(sum, x); // ymm0
    }    
    //add all the components of sum together
    // only needs AVX2
    return hsum_8x32(sum);
}

int main() {
    const int n = 1'000'000'000;
//    uint32_t * a = new uint32_t[n];
    uint32_t* a = (uint32_t*)aligned_alloc(32, n*sizeof(uint32_t));
    benchmark("scan_array", scan_array, a, n);
    benchmark("sum_array", sum_array, a, n);
    benchmark("sum_array2", sum_array2, a, n);
    cout << "sum_array3=" << sum_array3((__m256i*)a, n);
//    delete [] a;
    free(a);
}