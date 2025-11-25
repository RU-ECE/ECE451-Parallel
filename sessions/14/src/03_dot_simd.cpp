#include <immintrin.h>

float dot_simd(const float a[], const float b[], const int n) {
	auto sum = 0.0f;
#pragma omp simd safelen(16)
	for (auto i = 0; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}

float dot_simd2(const float a[], const float b[], const int n) {
	auto sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
	for (auto i = 0; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}


float dot_simd3(const float a[], const float b[], const int n) {
	auto sum = 0.0f;
#pragma omp parallel for simd reduction(+ : sum)
	for (auto i = 0; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}


float prod(const float a[], const int n) {
	auto prod = 1.0f;
#pragma omp parallel for reduction(* : prod)
	for (auto i = 0; i < n; i++)
		prod *= a[i];
	return prod;
}


float horizontal_sum(const __m256 v) {
	// Step 1: Perform a horizontal add of adjacent pairs
	__m256 temp = _mm256_hadd_ps(v, v); //  [A+B, C+D, E+F, G+H, A+B, C+D, E+F, G+H]

	// Step 2: Reduce further by swapping and adding across lanes
	const __m128 low = _mm256_castps256_ps128(temp); // Lower 128 bits: [A+B, C+D, E+F, G+H]
	const auto high = _mm256_extractf128_ps(temp, 1); // Upper 128 bits: [A+B, C+D, E+F, G+H]
	__m128 result = _mm_add_ps(low, high); // [A+B+C+D, E+F+G+H, unused, unused]

	// Step 3: Final horizontal add to get a single sum
	result = _mm_hadd_ps(result, result); // [A+B+C+D+E+F+G+H, unused, unused, unused]
	result = _mm_hadd_ps(result, result); // [A+B+C+D+E+F+G+H, unused, unused, unused]

	// Step 4: Extract the final sum
	return _mm_cvtss_f32(result);
}
/*
We can do better!
Let's do this by hand in avx2 intrinsics

*/
float dot_avx2manual(const float a[], const float b[], const int n) {
	__m256 sum = _mm256_setzero_ps();
	for (auto i = 0; i < n; i += 8)
		sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
	return horizontal_sum(sum);
}
