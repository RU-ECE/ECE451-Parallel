#include <chrono>
#include <immintrin.h>
#include <iostream>

using namespace std;
using namespace chrono;

void benchmark(const string& name, unsigned int (*f)(const unsigned int[], int), unsigned int a[], const int n) {
	const auto t0 = high_resolution_clock::now();
	f(a, n);
	const auto t1 = high_resolution_clock::now();
	const auto elapsed = duration_cast<milliseconds>(t1 - t0).count();
	cout << name << " elapsed=" << elapsed << "ms\n";
}

unsigned int scan_array(const unsigned int a[], const int n) {
	for (auto i = 0; i < n; i++)
		a[i];
	return 0;
}

unsigned int sum_array(const unsigned int a[], const int n) {
	auto sum = 0;
	for (auto i = 0; i < n; i++)
		sum += a[i];
	return sum;
}

//                   %rdi          %rsi
unsigned int sum_array2(const unsigned int a[], const int n) {
	auto sum = 0U;
	for (auto i = 0; i < n; i++)
		sum += a[i];
	return sum;
}
/*
0:   0x0000000000000000
8:   0x0000000000000000
16:  0x0000000000000000
24:  0x0000000000000000
32:  0x0000000000000000
40:  0x0000000000000000
48:  0x0000000000000000
56:  0x0000000000000000
64:  0x0000000000000000


*/

/*
	AVX2 registers

	ymm0
	ymm1
	ymm2
	ymm3
	ymm4
	ymm5
	ymm6
	ymm7
	...
	ymm15
*/
unsigned int hsum_epi32_avx(const __m128i x) {
	const auto hi64 =
		_mm_unpackhi_epi64(x, x); // 3-operand non-destructive AVX lets us save a byte without needing a movdqa
	auto sum64 = _mm_add_epi32(hi64, x);
	const auto hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1)); // Swap the low two elements
	const auto sum32 = _mm_add_epi32(sum64, hi32);
	return _mm_cvtsi128_si32(sum32); // movd
}

unsigned int hsum_8x32(__m256i v) {
	const auto sum128 = _mm_add_epi32(
		_mm256_castsi256_si128(v),
		_mm256_extracti128_si256(v, 1)); // silly GCC uses a longer AXV512VL instruction if AVX512 is enabled :/
	return hsum_epi32_avx(sum128);
}

unsigned int sum_array3(__m256i a[], const int n) {
	auto sum = _mm256_setzero_si256(); // ymm0
	for (auto i = 0; i < n; i += 8) {
		const auto x = _mm256_loadu_si256(&a[i]); // ymm1?
		sum = _mm256_add_epi32(sum, x); // ymm0
	}
	// add all the components of sum together
	//  only needs AVX2
	return hsum_8x32(sum);
}

int main() {
	constexpr auto n = 1'000'000'000;
	//    uint32_t * a = new uint32_t[n];
	const auto a = static_cast<unsigned int*>(aligned_alloc(32, n * sizeof(unsigned int)));
	benchmark("scan_array", scan_array, a, n);
	benchmark("sum_array", sum_array, a, n);
	benchmark("sum_array2", sum_array2, a, n);
	cout << "sum_array3=" << sum_array3(reinterpret_cast<__m256i*>(a), n);
	//    delete [] a;
	free(a);
}
