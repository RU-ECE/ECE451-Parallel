#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <malloc.h>
#include <memory.h>

using namespace std;
using namespace chrono;

void product(const float* a, const float* b, float* c, const int n) {
	for (uint32_t i = 0; i < n; i += 8) { // think in terms of floats, we do 0..7, 8..15, ...
		__m256 tempa = _mm256_loadu_ps(a + i); // first time a+0 = a, second time a + 8*sizeof(float)
		const __m256 tempb = _mm256_loadu_ps(b + i);
		tempa = _mm256_mul_ps(tempa, tempb);
		_mm256_storeu_ps(c + i, tempa);
	}
}

float sum(const float* a, const int n) {
	float sum = 0;
	for (auto i = 0; i < n; i++)
		sum += a[i];
	return sum;
}

float dotproduct_vector(const float* a, const float* b, const int n) {
	__m256 sum = _mm256_setzero_ps(); // set all 8 floats to 0
	for (auto i = 0; i < n; i += 8) {
		__m256 tempa = _mm256_loadu_ps(a + i); // first time a+0 = a, second time a + 8*sizeof(float)
		const __m256 tempb = _mm256_loadu_ps(b + i);
		tempa = _mm256_mul_ps(tempa, tempb);
		sum = _mm256_add_ps(sum, tempa);
	}
	// horizontal sum using vector instructions
	const __m256 temp1 = _mm256_hadd_ps(sum, sum);
	__m256 temp2 = _mm256_hadd_ps(temp1, temp1);
	const __m128 low = _mm256_castps256_ps128(temp2);
	const auto high = _mm256_extractf128_ps(temp2, 1);
	const __m128 final_sum = _mm_add_ps(low, high);
	const float fsum = _mm_cvtss_f32(final_sum);
	return fsum;
}

float dotproduct_scalar(const float* a, const float* b, const int n) {
	float sum = 0;
	for (auto i = 0; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}

int main() {
	constexpr uint64_t n = 1024 * 1024 * 128;
	const auto a = static_cast<float*>(aligned_alloc(32, n * 32));
	const auto b = static_cast<float*>(aligned_alloc(32, n * 32));
	const auto c = static_cast<float*>(aligned_alloc(32, n * 32));
	for (auto i = 0; i < n; i++) {
		a[i] = 1;
		b[i] = 1;
	}
	for (auto i = 0; i < n; i += 3) {
		a[i] = 3;
		b[i] = 1;
	}
	auto t0 = high_resolution_clock().now();
	product(a, b, c, n);
	const float s = sum(c, n);
	auto t1 = high_resolution_clock().now();
	duration<double> elapsed_seconds = t1 - t0;

	// Output the time
	cout << "2 passes summation time: " << elapsed_seconds.count() << " seconds\n";
	cout << s << endl;

	t0 = high_resolution_clock().now();
	float dot = dotproduct_scalar(a, b, n);
	t1 = high_resolution_clock().now();
	elapsed_seconds = t1 - t0;
	cout << dot << endl;

	// Output the time
	cout << "dot product scalar time: " << elapsed_seconds.count() << " seconds\n";

	t0 = high_resolution_clock().now();
	dot = dotproduct_vector(a, b, n);
	t1 = high_resolution_clock().now();
	elapsed_seconds = t1 - t0;
	cout << dot << endl;

	// Output the time
	cout << "dot product scalar time: " << elapsed_seconds.count() << " seconds\n";


	free(a);
	free(b);
	free(c);
}
