#include <chrono>
#include <immintrin.h>
#include <iostream>

using namespace std;
using namespace chrono;

double dot_product(const double a[], const double b[], const int n) {
	auto sum = 0.0;
	for (auto i = 0; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}

double dot_product(const __m256d a[], const __m256d b[], const int n) {
	auto sum = _mm256_set1_pd(0);
	for (auto i = 0; i < n / 4; i++) {
		const auto temp = _mm256_mul_pd(a[i], b[i]);
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
	constexpr auto n = 1024 * 1024;
	const auto x = static_cast<double*>(aligned_alloc(sizeof(__m256d), n * sizeof(double)));
	const auto y = static_cast<double*>(aligned_alloc(sizeof(__m256d), n * sizeof(double)));
	for (auto i = 0; i < n; i++) {
		x[i] = i;
		y[i] = i;
	}
	auto t0 = high_resolution_clock::now();
	const auto s1 = dot_product(x, y, n);
	auto t1 = high_resolution_clock::now();
	auto elapsed = duration_cast<microseconds>(t1 - t0);
	cout << "s1=" << s1 << "\telapsed: " << elapsed.count() << "usec\n";

	t0 = high_resolution_clock::now();
	const auto s2 = dot_product(reinterpret_cast<const __m256d*>(x), reinterpret_cast<const __m256d*>(y), n);
	t1 = high_resolution_clock::now();
	elapsed = duration_cast<microseconds>(t1 - t0);
	cout << "s2=" << s2 << "\telapsed: " << elapsed.count() << "usec\n";
}
