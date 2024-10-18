#include <iostream>
#include <chrono>
#include <immintrin.h>
using namespace std;

/*
    when you read arr[0] (32-bit) the computer reads 64-bit from RAM
    when you read arr[1] that's already in cache from the first read

    CAS-RAS-precharge
    46-45-45
    CAS: if you are in a row, it takes you 46 clocks to get a memory location
    RAS: if you aren't row, it takes 45+46

    a[i] access
    1. is it in L1 cache? if so 2clocks
    2. is it in L2 cache? if so 3-5clocks
    3. is it in L3 cache? if so 6-8clocks
    4. is it in RAM? if you are already in the row 46
    5. if not 46+45
    remember, MOTHERBOARD + CPU and MMU will add delays on top

    DDR4 RAM has 8-burst, and I believe 4-cancel
    DDR5 RAM has 16-burst
    2 memory banks operate independently

    sequential access 46 + 15*1 = 61
*/

uint32_t sum_scalar(uint32_t *arr, uint32_t n) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

uint32_t parallel_sum_avx2(uint32_t *arr, uint32_t n) {
    __m256i sum = _mm256_setzero_si256();
    for (uint32_t i = 0; i < n; i+=8) {
        sum = _mm256_add_epi32(sum, _mm256_loadu_si256((__m256i*)&arr[i]));
    }
 // Horizontal summation of the 8 packed 32-bit integers
    __m128i low = _mm256_castsi256_si128(sum);       // Get the low 128 bits
    __m128i high = _mm256_extracti128_si256(sum, 1); // Get the high 128 bits
    __m128i result128 = _mm_add_epi32(low, high);    // Sum the two 128-bit halves

    // Sum the 4 elements in the 128-bit register
    result128 = _mm_hadd_epi32(result128, result128);
    result128 = _mm_hadd_epi32(result128, result128);

    // Extract the final sum
    return _mm_cvtsi128_si32(result128);    
//    return _mm512_reduce_add_epi32(sum);
//    return _mm256_reduce_add_epi32(sum); // check out this cool convenience function!
}

float dot_product_scalar(float *a, float *b, uint32_t n) {
    float sum = 0;
    for (uint32_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}


void vec_product_scalar(float *a, float *b, float* c, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}


float vec_product_scalar2(float *a, float *b, float* c, uint32_t n) {
    float sum = 0;
    for (uint32_t i = 0; i < n; i++) {
        sum += c[i] + a[i] * b[i];
    }
    return sum;
}

float dot_product_avx2(float *a, float *b, uint32_t n) {
    __m256 sum = _mm256_setzero_ps();
    for (uint32_t i = 0; i < n; i++) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
    }
     // Horizontal summation of the 8 packed 32-bit integers
    __m128i low = _mm256_castsi256_si128((__m256i)sum);       // Get the low 128 bits
    __m128i high = _mm256_extracti128_si256(sum, 1); // Get the high 128 bits
    __m128i result128 = _mm_add_epi32(low, high);    // Sum the two 128-bit halves

    // Sum the 4 elements in the 128-bit register
    result128 = _mm_hadd_epi32(result128, result128);
    result128 = _mm_hadd_epi32(result128, result128);

    // Extract the final sum
    return _mm_cvtsi128_si32(result128);    
}

// sum the reciprocals 1/1 + 1/2 + 1/3 + ...1/n
float sumit(uint32_t n) {
    float sum = 0;
    for (uint32_t i = 1; i <= n; i++) {
        sum += 1.0f / i;
    }
    return sum;
}

float vector_sumit(float *arr, uint32_t n) {
    __m256f sum = _mm256_setzero_ps();
    const float k[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const float eightarr[] = {8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f};
    __m256f eightarr_vec = _mm256_loadu_ps(eightarr);
    __m256f eight = _m256_broadcast_ps(8.0f); // 8,8,8,8,8,8,8,8
    __m256f one = _mm256_broadcast_ps(1.0f); // 1,1,1,1,1,1,1,1
    __m256f delta = _mm256_loadu_ps(k);
    // 1 2 3 4 5 6 7 8     9 10 11 12 13 14 15 16
    for (uint32_t i = 0; i < n; i+= 8) {
        sum = _mm256_add_ps(sum, _mm256_mul_ps(one, delta)); // 1/1, 1/2, 1/3 ... 1/8
        delta = _mm256_add_ps(delta, eightarr_vec); // 9,10,11,12,13,14,15,16
    }

    // horizontal summation
    return sum;
}


// do a sorting network for groups of 8 elements
void sort8_scalar(uint32_t *arr, uint32_t n) {
    for (uint32_t i = 0; i < n; i+= 8) {
        uint32_t a = arr[i];
        uint32_t b = arr[i + 1];
        uint32_t c = arr[i + 2];
        uint32_t d = arr[i + 3];
        uint32_t e = arr[i + 4];
        uint32_t f = arr[i + 5];
        uint32_t g = arr[i + 6];
        uint32_t h = arr[i + 7];
        
    }
}

int main() {
    uint32_t n = 8*50'000'000; // why can n not be 100? Must be multiple of 8!
    uint32_t* p = (uint32_t*)aligned_alloc(32, n*4);

    auto t0 = std::chrono::high_resolution_clock::now();
    uint32_t s = sum_scalar(p, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    cout << duration << " sum = " << s << endl;

    t0 = std::chrono::high_resolution_clock::now();
    uint32_t p_s = parallel_sum_avx2(p, n);
    t1 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    cout << duration << " sum = " << p_s << endl;

    float* p2 = (float*)p;
    t0 = std::chrono::high_resolution_clock::now();
    float d = dot_product_scalar(p2, p2, n);
    t1 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    cout << duration << " sum = " << d << endl;

    t0 = std::chrono::high_resolution_clock::now();
    float d2 = dot_product_avx2(p2, p2, n);
    t1 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    cout << duration << " sum = " << d2 << endl;


    free(p);
    return 0;
}

