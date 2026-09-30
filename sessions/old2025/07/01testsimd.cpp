#include <iostream>
#include <cstdint>
#include <chrono>
#include <immintrin.h>
#include <malloc.h>
#include <memory.h>
using namespace std;

void product(float* a, float* b, float* c, int n) {
    for (uint32_t i = 0; i < n; i+=8) { // think in terms of floats, we do 0..7, 8..15, ...
      __m256 tempa = _mm256_loadu_ps(a+i); // first time a+0 = a, second time a + 8*sizeof(float)
      __m256 tempb = _mm256_loadu_ps(b+i);
      tempa = _mm256_mul_ps(tempa, tempb);
      _mm256_storeu_ps(c+i, tempa);
    }
}

float sum(float* a, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

float dotproduct_vector(float* a, float* b, int n) {
    __m256 sum = _mm256_setzero_ps(); // set all 8 floats to 0
    for (int i = 0; i < n; i+=8) {
        __m256 tempa = _mm256_loadu_ps(a+i); // first time a+0 = a, second time a + 8*sizeof(float)
        __m256 tempb = _mm256_loadu_ps(b+i);
        tempa = _mm256_mul_ps(tempa, tempb);
        sum = _mm256_add_ps(sum, tempa);
    }
    // horizontal sum using vector instructions
    __m256 temp1 = _mm256_hadd_ps(sum, sum);
    __m256 temp2 = _mm256_hadd_ps(temp1, temp1);
    __m128 low = _mm256_castps256_ps128(temp2);
    __m128 high = _mm256_extractf128_ps(temp2, 1);
    __m128 final_sum = _mm_add_ps(low, high);
    float fsum = _mm_cvtss_f32(final_sum);
    return fsum;
}

float dotproduct_scalar(float* a, float* b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

int main() {
    const uint64_t n = 1024*1024*128;
    float* a = (float*)aligned_alloc(32, n*32);
    float* b = (float*)aligned_alloc(32, n*32);
    float* c = (float*)aligned_alloc(32, n*32);
    for (int i = 0; i < n; i++) {
        a[i] = 1;
        b[i] = 1;
    }
    for (int i = 0; i < n; i+=3) {
        a[i] = 3;
        b[i] = 1;
    }
    auto t0 = chrono::high_resolution_clock().now();
    product(a, b, c, n);
    float s = sum(c, n);
    auto t1 = chrono::high_resolution_clock().now();
     std::chrono::duration<double> elapsed_seconds = t1 - t0;
    
    // Output the time
    std::cout << "2 passes summation time: " << elapsed_seconds.count() << " seconds\n";
    cout << s << '\n';

    t0 = chrono::high_resolution_clock().now();
    float dot = dotproduct_scalar(a, b, n);
    t1 = chrono::high_resolution_clock().now();
    elapsed_seconds = t1 - t0;
    cout << dot << '\n';

    // Output the time
    std::cout << "dot product scalar time: " << elapsed_seconds.count() << " seconds\n";

    t0 = chrono::high_resolution_clock().now();
    dot = dotproduct_vector(a, b, n);
    t1 = chrono::high_resolution_clock().now();
    elapsed_seconds = t1 - t0;
   cout << dot << '\n';
 
    // Output the time
    std::cout << "dot product scalar time: " << elapsed_seconds.count() << " seconds\n";


    free(a);
    free(b);
    free(c);

}