#include <immintrin.h>

using namespace std;

void memcopy1(char* dest, const char* src, unsigned long num_bytes) {
	// firsrt memory cost RAS + CAS + 15  =  16 reads + CAS + 15 = 16 reads
	// m0 m1 m2 m3 m4 m5 m6 m7 m8 m9 ...
	for (; num_bytes > 0; num_bytes--) {
		*dest = *src;
		dest++;
		src++;
	}
}
// assume num_bytes is an even multiple of 8
void memcopy2(unsigned long* dest, const unsigned long* src, unsigned long num_bytes) {
	for (; num_bytes > 0; num_bytes -= 8) {
		*dest = *src;
		dest++;
		src++;
	}
}

// vectorized memcopy// assume num_bytes is an even multiple of 8
void memcopy3(__m256i* dest, const __m256i* src, unsigned long num_bytes) {
	for (; num_bytes > 0; num_bytes -= 32) {
		const auto temp = _mm256_loadu_si256(src);
		_mm256_storeu_si256(dest, temp);
		dest++; // advances by 32 bytes
		src++;
	}
}

// read in 16 words because that is burst size on DDR5
// benchmarkvectorized memcopy// assume num_bytes is an even multiple of 8
void memcopy4(__m256i* dest, const __m256i* src, unsigned long num_bytes) {
	for (; num_bytes > 0; num_bytes -= 128) {
		const auto a = _mm256_loadu_si256(src);
		const auto b = _mm256_loadu_si256(src + 1);
		const auto c = _mm256_loadu_si256(src + 2);
		const auto d = _mm256_loadu_si256(src + 3);
		_mm256_storeu_si256(dest, a);
		_mm256_storeu_si256(dest + 1, b);
		_mm256_storeu_si256(dest + 2, c);
		_mm256_storeu_si256(dest + 3, d);
		dest += 4; // advances by 32 bytes
		src += 4;
	}
}
