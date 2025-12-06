#include <chrono>
#include <immintrin.h>
#include <iostream>

using namespace std;
using namespace chrono;

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

unsigned int sum_scalar(const unsigned int* arr, const unsigned int n) {
	auto sum = 0U;
	for (auto i = 0U; i < n; i++)
		sum += arr[i];
	return sum;
}

unsigned int parallel_sum_avx2(unsigned int* arr, const unsigned int n) {
	auto sum = _mm256_setzero_si256();
	for (auto i = 0U; i < n; i += 8)
		sum = _mm256_add_epi32(sum, _mm256_loadu_si256(reinterpret_cast<__m256i*>(&arr[i])));
	// Horizontal summation of the 8 packed 32-bit integers
	const auto low = _mm256_castsi256_si128(sum); // Get the low 128 bits
	const auto high = _mm256_extracti128_si256(sum, 1); // Get the high 128 bits
	auto result128 = _mm_add_epi32(low, high); // Sum the two 128-bit halves
	// Sum the 4 elements in the 128-bit register
	result128 = _mm_hadd_epi32(result128, result128);
	result128 = _mm_hadd_epi32(result128, result128);
	// Extract the final sum
	return _mm_cvtsi128_si32(result128);
	// return _mm512_reduce_add_epi32(sum);
	// return _mm256_reduce_add_epi32(sum); // check out this cool convenience function!
}

float dot_product_scalar(const float* a, const float* b, const unsigned int n) {
	auto sum = 0.0f;
	for (unsigned int i = 0; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}

void vec_product_scalar(const float* a, const float* b, float* c, const unsigned int n) {
	for (auto i = 0U; i < n; i++)
		c[i] = a[i] * b[i];
}

float vec_product_scalar2(const float* a, const float* b, const float* c, const unsigned int n) {
	auto sum = 0.0f;
	for (auto i = 0U; i < n; i++)
		sum += c[i] + a[i] * b[i];
	return sum;
}

float dot_product_avx2(const float* a, const float* b, const unsigned int n) {
	auto sum = _mm256_setzero_ps();
	for (auto i = 0U; i < n; i++) {
		const auto a_vec = _mm256_loadu_ps(&a[i]), b_vec = _mm256_loadu_ps(&b[i]);
		sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
	}
	// Horizontal summation of the 8 packed 32-bit integers
	const auto low = _mm256_castsi256_si128(reinterpret_cast<__m256i>(sum)); // Get the low 128 bits
	const auto high = _mm256_extracti128_si256(sum, 1); // Get the high 128 bits
	auto result128 = _mm_add_epi32(low, high); // Sum the two 128-bit halves
	// Sum the 4 elements in the 128-bit register
	result128 = _mm_hadd_epi32(result128, result128);
	result128 = _mm_hadd_epi32(result128, result128);
	// Extract the final sum
	return static_cast<float>(_mm_cvtsi128_si32(result128));
}

// sum the reciprocals 1/1 + 1/2 + 1/3 + ...1/n
float sumit(const unsigned int n) {
	auto sum = 0.0f;
	for (auto i = 1U; i <= n; i++)
		sum += 1.0f / static_cast<float>(i);
	return sum;
}

float vector_sumit(float* arr, const unsigned int n) {
	auto sum = _mm256_setzero_ps();
	constexpr float k[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	constexpr float eightarr[] = {8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f};
	const auto eightarr_vec = _mm256_loadu_ps(eightarr);
	const auto eight = _mm256_set1_ps(8.0f); // broadcast scalar 8.0f
	const auto one = _mm256_set1_ps(1.0f); // broadcast scalar 1.0f
	auto delta = _mm256_loadu_ps(k);
	for (auto i = 0U; i < n; i += 8) {
		// sum += 1.0f / delta (reciprocals)
		sum = _mm256_add_ps(sum, _mm256_div_ps(one, delta));
		delta = _mm256_add_ps(delta, eightarr_vec);
	}
	// horizontal sum of the 8 floats in 'sum'
	const auto low = _mm256_castps256_ps128(sum);
	const auto high = _mm256_extractf128_ps(sum, 1);
	auto s128 = _mm_add_ps(low, high);
	s128 = _mm_hadd_ps(s128, s128);
	s128 = _mm_hadd_ps(s128, s128);
	return _mm_cvtss_f32(s128);
}

// do a sorting network for groups of 8 elements
void sort8_scalar(const unsigned int* arr, const unsigned int n) {
	for (auto i = 0U; i < n; i += 8) {
		auto a = arr[i], b = arr[i + 1], c = arr[i + 2], d = arr[i + 3], e = arr[i + 4], f = arr[i + 5], g = arr[i + 6],
			 h = arr[i + 7];
	}
}

int main() {
	constexpr auto n = 8U * 50'000'000; // why can n not be 100? Must be multiple of 8!
	const auto p = static_cast<unsigned int*>(aligned_alloc(32, n * 4));
	auto t0 = high_resolution_clock::now();
	const auto s = sum_scalar(p, n);
	auto t1 = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(t1 - t0).count();
	cout << duration << " sum = " << s << endl;
	t0 = high_resolution_clock::now();
	const auto p_s = parallel_sum_avx2(p, n);
	t1 = high_resolution_clock::now();
	duration = duration_cast<microseconds>(t1 - t0).count();
	cout << duration << " sum = " << p_s << endl;
	const auto p2 = reinterpret_cast<float*>(p);
	t0 = high_resolution_clock::now();
	const auto d = dot_product_scalar(p2, p2, n);
	t1 = high_resolution_clock::now();
	duration = duration_cast<microseconds>(t1 - t0).count();
	cout << duration << " sum = " << d << endl;
	t0 = high_resolution_clock::now();
	const float d2 = dot_product_avx2(p2, p2, n);
	t1 = high_resolution_clock::now();
	duration = duration_cast<microseconds>(t1 - t0).count();
	cout << duration << " sum = " << d2 << endl;
	free(p);
	return 0;
}
