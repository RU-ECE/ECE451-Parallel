#include <immintrin.h>
#include <iostream>

using namespace std;

ostream& operator<<(ostream& s, __m256i a) {
	auto temp = _mm256_extract_epi64(a, 3);
	s << (temp >> 32) << " " << (temp & 0xFFFFFFFF) << ' ';
	temp = _mm256_extract_epi64(a, 2);
	s << (temp >> 32) << " " << (temp & 0xFFFFFFFF) << ' ';
	temp = _mm256_extract_epi64(a, 1);
	s << (temp >> 32) << " " << (temp & 0xFFFFFFFF) << ' ';
	temp = _mm256_extract_epi64(a, 0);
	s << (temp >> 32) << " " << (temp & 0xFFFFFFFF) << ' ';
	return s;
}

void printvadd(const __m256i a, const __m256i b) {
	const auto c = _mm256_add_epi32(a, b);
	cout << c;
}

void printvadd_andload(const unsigned int* ap, const unsigned int* bp) {
	const auto a = _mm256_load_si256(reinterpret_cast<__m256i const*>(ap)),
			   b = _mm256_load_si256(reinterpret_cast<__m256i const*>(bp)), c = _mm256_add_epi32(a, b);
	cout << c;
}

int main();
