#include <immintrin.h>
#include <iostream>

using namespace std;

/*
 * uint32_t = 32 bits
 * uint64_t = 64 bits
 * float = 32 bits
 * double = 64 bits
 *
 * %xmm0 .. %xmm15   SSE (128 bit), AVX (128 bit), 2003
 * %ymm0 .. %ymm15   AVX2 (256 bit), 2008
 * %zmm0 .. %zmm31   AVX512 (512 bit), 2018
 *
 * AVX512 is known to be a power hog and will reduce your processes clock speed so only worth it if you REALLY USE IT
 *
 * NON-PORTABLE
 *
 * Adobe will have plugins to use various architectures
 *
 * future?
 *
 * ARM scalable supercomputer vector extensions
 */

double sum(const double a[], const double b[], const int n) {
	auto sum = 0.0;
	for (auto i = 0; i < n; i++)
		sum += a[i] + b[i];
	return sum;
}

// this function will assume that n is a multiple of 4
double sum(const __m256d a[], const __m256d b[], const int n) {
	__m256d sum = {0, 0, 0, 0};
	for (auto i = 0; i < n; i += 4) {
		const auto temp = _mm256_add_pd(a[i], b[i]);
		sum = _mm256_add_pd(sum, temp);
	}
	// sum = (s1, s2, s3, s4)
	// sum = (__, __, s1+s3, s2+s4)
	// sum = (__, ___, ____, s1+s2+s3+s4)
	// return low part of sum
	// perform a horizontal add to get a single sum
	const auto s = sum[0] + sum[1] + sum[2] + sum[3];
	return s;
}

int main() {
	constexpr auto a = 1UL, // mov $1, %rax
		b = 2UL;
	auto c = a + b; // add $2, %rax
	// ARM: add X5, X0, X1   // X5 = X0 + X1

	{
		constexpr auto x = 1.5; //%xmm0
		constexpr auto y = 2.3; //%xmm1
		auto z = x + y; //%xmm2
	}

	// in SIMD we use a single instruction to perform MULTIPLE Operation
	auto a1 = _mm_set1_epi32(1); // 4 bytes %xmm
	auto a2 = _mm256_set1_epi32(1); // 8 bytes %ymm
	// I don't have AVX512!    __m512i a3 = _mm512_set1_epi32(1); // 16 bytes %zmm
	//     __m256d _mm256_add_pd (__m256d a, __m256d b) // 4 double
	//__m256 _mm256_add_ps (__m256 a, __m256 b) // 8 float
	constexpr auto n = 1024;
	const auto x = static_cast<double*>(aligned_alloc(sizeof(__m256d), 1024 * sizeof(double)));
	const auto y = static_cast<double*>(aligned_alloc(sizeof(__m256d), 1024 * sizeof(double)));
	//    double* x = new double[n];
	//    double* y = new double[n];
	for (auto i = 0; i < n; i++) {
		x[i] = i;
		y[i] = i;
	}
	const auto s1 = sum(x, y, n);
	const auto s2 = sum(reinterpret_cast<__m256d*>(x), reinterpret_cast<__m256d*>(x), n);
	cout << "s1=" << s1 << endl;
	cout << "s2=" << s2 << endl;
}
