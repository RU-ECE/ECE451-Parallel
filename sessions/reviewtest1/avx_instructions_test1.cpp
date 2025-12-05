#include <iostream>
#include <chrono>
#include <immintrin.h>

#ifdef RANDOMVALUE
#include <random>
#include <omp.h>
#endif
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

float reduce_add_ps(__m256 a) {
    /**
     * source: https://stackoverflow.com/questions/26896432/horizontal-add-with-m512-avx512
     * documentation: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX_ALL,AVX_512
     * a: [f1, f2, ..., f8]
     * t1: [f1+f2, f3+f4, f1+f2, f3+f4, f5+f6, f7+f8, f5+f6, f7+f8]
     * t2: [f1+f2+f3+f4, ...(2 times), f1+f2+f3+f4, f5+f6+f7+f8, ...(2 times), f5+f6+f7+f8]
     * low: [f5+f6+f7+f8, ...(total 4 times)]
     * high: [f1+f2+f3+f4, ...(total 4 times)]
     * _mm_and_ss: A 128-bit vector of [4 x float] whose lower 32 bits contain the sum of the lower 32 bits of both
     *             operands. The upper 96 bits are copied from the upper 96 bits of the first source operand.
     * result128: [f5+f6+f7+f8, f5+f6+f7+f8, f5+f6+f7+f8, f1+f2+f3+f4+f5+f6+f7+f8]
     * _mm_cvtss_f32: extract lower 32 bits
     */
    __m256 t1 = _mm256_hadd_ps(a, a);
    __m256 t2 = _mm256_hadd_ps(t1, t1);
    __m128 low = _mm256_castps256_ps128(t2);    // Get the low 128 bits
    __m128 high = _mm256_extractf128_ps(t2, 1);    // Get the high 128 bits
    __m128 result128 = _mm_add_ss(low, high);// Sum the two 128-bit halves

    // Extract the final sum
    return _mm_cvtss_f32(result128);
}

int reduce_add_epi32(__m256i a) {
    /**
     * I assume doing horizontal addition on __m256i is faster than on __m128i because later only use half of 256-register
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
     */

    /**
     * a: [d1, d2, ..., d8]
     * t1: [d1+d2, d3+d4, d1+d2, d3+d4, d5+d6, d7+d8, d5+d6, d7+d8]
     * t2: [d1+d2+d3+d4, ...(2 times), d1+d2+d3+d4, d5+d6+d7+d8, ...(2 times), d5+d6+d7+d8]
     * low: [d5+d6+d7+d8, ...(total 4 times)]
     * high: [d1+d2+d3+d4, ...(total 4 times)]
     * result128: [d1+d2+d3+d4+d5+d6+d7+d8, ...(total 4 times)]
     */
    __m256i t1 = _mm256_hadd_epi32(a, a);
    __m256i t2 = _mm256_hadd_epi32(t1, t1);
    __m128i low = _mm256_castsi256_si128(t2);    // Get the low 128 bits
    __m128i high = _mm256_extracti128_si256(t2, 1);    // Get the high 128 bits
    __m128i result128 = _mm_add_epi32(low, high);// Sum the two 128-bit halves

    // Extract the final sum
    return _mm_cvtsi128_si32(result128);
}

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

    return reduce_add_epi32(sum);
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
    for (uint32_t i = 0; i < n; i+=8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
    }

    return reduce_add_ps(sum);
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
    __m256 sum = _mm256_setzero_ps();
    const float k[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const float eightarr[] = {8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f};
    __m256 eightarr_vec = _mm256_loadu_ps(eightarr);
    /*
     * _mm256_broadcast_ps: Broadcast 128 bits from memory (composed of 4 packed single-precision (32-bit) floating-point elements) to all elements of dst.
     *          tmp[127:0] := MEM[mem_addr+127:mem_addr]
     *          dst[127:0] := tmp[127:0]
     *          dst[255:128] := tmp[127:0]
     *          dst[MAX:256] := 0
     * _mm256_broadcast_ss: Broadcast a single-precision (32-bit) floating-point element from memory to all elements of dst.
     *          tmp[31:0] := MEM[mem_addr+31:mem_addr]
     *          FOR j := 0 to 7
     *              i := j*32
     *              dst[i+31:i] := tmp[31:0]
     *          ENDFOR
     *          dst[MAX:256] := 0
     */
    const float eight_f = 8.0f;
    const float one_f = 1.0f;
    __m256 eight = _mm256_broadcast_ss(&eight_f); // 8,8,8,8,8,8,8,8
    __m256 one = _mm256_broadcast_ss(&one_f); // 1,1,1,1,1,1,1,1
    __m256 delta = _mm256_loadu_ps(k);
    // 1 2 3 4 5 6 7 8     9 10 11 12 13 14 15 16
    for (uint32_t i = 0; i < n; i += 8) {
        sum = _mm256_add_ps(sum, _mm256_div_ps(one, delta)); // 1/1, 1/2, 1/3 ... 1/8
        delta = _mm256_add_ps(delta, eightarr_vec); // 9,10,11,12,13,14,15,16
    }

    return reduce_add_ps(sum);
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
    uint32_t n = 8*50000000;    // why can n not be 100? Must be multiple of 8!
    uint32_t* p = (uint32_t*)aligned_alloc(32, n*4);

#ifdef RANDOMVALUE
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> disInt(-5, 5);

    #pragma omp parallel for
    for (uint32_t i = 0; i < n; ++i) {
        p[i] = disInt(gen);
    }
#endif

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

#ifdef RANDOMVALUE
    uniform_real_distribution<float> disFloat(-2.0f, 2.0f);

    #pragma omp parallel for
    for (uint32_t i = 0; i < n; ++i) {
        p2[i] = disFloat(gen);
    }
#endif

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

