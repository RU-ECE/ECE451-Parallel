#include <ctime>
#include <immintrin.h>
#include <iostream>
#include <memory>

using namespace std;

float dot(const float x[], const float y[], const int n) {
	float sum = 0;
	for (auto i = 0; i < n; i++)
		sum += x[i] * y[i];
	return sum;
}

float dot_product_avx2(const float* a, const float* b, const int n) {
	auto result = _mm256_setzero_ps();

	for (auto i = 0; i < n; i += 8)
		result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));

	// Horizontal addition of elements in the result vector
	const auto hi128 = _mm256_extractf128_ps(result, 1);
	auto dotproduct = _mm_add_ps(_mm256_castps256_ps128(result), hi128);
	dotproduct = _mm_hadd_ps(dotproduct, dotproduct);
	dotproduct = _mm_hadd_ps(dotproduct, dotproduct);

	float dot_product;
	_mm_store_ss(&dot_product, dotproduct);

	return dot_product;
}

void init(float* a, const float value, const int n) {
	for (auto i = 0; i < n; i++)
		a[i] = value;
}

int main() {
	constexpr auto n = 100'000'000;
	const auto a = make_unique<float[]>(n);
	const auto b = make_unique<float[]>(n);
	auto t0 = clock();
	init(a.get(), 1.5, n);
	init(b.get(), 2.3, n);
	auto t1 = clock();
	cout << "init: " << (t1 - t0) << endl;

	t0 = clock();
	cout << dot(a.get(), b.get(), n) << endl;
	t1 = clock();
	cout << "elapsed time dot: " << (t1 - t0) << endl;

	t0 = clock();
	cout << dot_product_avx2(a.get(), b.get(), n) << endl;
	t1 = clock();
	cout << "elapsed time dot_product_avx2: " << (t1 - t0) << endl;
}
