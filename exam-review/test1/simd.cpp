#include <immintrin.h>
#include <iostream>

using namespace std;

__m256d sum(const double x[], const int n) {
	auto s = _mm256_setzero_pd(); // four zeros
	for (auto i = 0; i < n; i += 4) {
		const auto vx = _mm256_loadu_pd(&x[i]); // load 4 doubles (unaligned)
		s = _mm256_add_pd(s, vx);
	}
	return s;
}

int main() { double x[] = {1, 2, 3, 4, 5, 6, 7, 8}; }
