#include <iostream>
#include <immintrin.h>
#include <time.h>
using namespace std;


// load the intel intrinsics

float dot(const float x[], const float y[], int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}


float dot_product_avx2(float* a, float* b, int n) {
    __m256 result = _mm256_setzero_ps();

    for (int i = 0; i < n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        result = _mm256_add_ps(result, _mm256_mul_ps(a_vec, b_vec));
    }

    // Horizontal addition of elements in the result vector
    __m128 hi128 = _mm256_extractf128_ps(result, 1);
    __m128 dotproduct = _mm_add_ps(_mm256_castps256_ps128(result), hi128);
    dotproduct = _mm_hadd_ps(dotproduct, dotproduct);
    dotproduct = _mm_hadd_ps(dotproduct, dotproduct);

    float dot_product;
    _mm_store_ss(&dot_product, dotproduct);
    
    return dot_product;
}

void init(float* a, float value, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = value;
    }
}

int main() {
    const int n = 1'000'000'00;
    float* a = new float[n];
    float* b = new float[n];
    clock_t t0 = clock();
    init(a, 1.5, n);
    init(b, 2.3, n);
    clock_t t1 = clock();
    cout << "init: " << (t1 - t0) << endl;

    t0 = clock();
    cout << dot(a, b, n) << endl;
    t1 = clock();
   cout << "elapsed time dot: " << (t1 - t0) << endl;

    t0 = clock();
    cout << dot_product_avx2(a, b, n) << endl;
    t1 = clock();
    cout << "elapsed time dot_product_avx2: " << (t1 - t0) << endl;
}