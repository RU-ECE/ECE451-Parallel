#include <iostream>
#include <immintrin.h>
#include <time.h>
using namespace std;


// load the intel intrinsics

float dot(const float x[], const float y[], const int n) {
    float sum = 0;
    for (auto i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}


float dot_product_avx2(float* a, float* b, const int n) {
    __m256 result = _mm256_setzero_ps();

    for (auto i = 0; i < n; i += 8) {
		const __m256 a_vec = _mm256_loadu_ps(&a[i]);
		const __m256 b_vec = _mm256_loadu_ps(&b[i]);
        result = _mm256_add_ps(result, _mm256_mul_ps(a_vec, b_vec));
    }

    // Horizontal addition of elements in the result vector
	const auto hi128 = _mm256_extractf128_ps(result, 1);
    __m128 dotproduct = _mm_add_ps(_mm256_castps256_ps128(result), hi128);
    dotproduct = _mm_hadd_ps(dotproduct, dotproduct);
    dotproduct = _mm_hadd_ps(dotproduct, dotproduct);

    float dot_product;
    _mm_store_ss(&dot_product, dotproduct);
    
    return dot_product;
}

void init(float* a, const float value, const int n) {
    for (auto i = 0; i < n; i++) {
        a[i] = value;
    }
}

int main() {
	constexpr auto n = 1'000'000'00;
	const auto a = new float[n];
	const auto b = new float[n];
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