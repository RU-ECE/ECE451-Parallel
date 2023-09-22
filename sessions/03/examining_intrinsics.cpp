#include <iostream>
#include <immintrin.h>

using namespace std;

ostream& operator <<(ostream& s, __m256i a) {
    uint64_t temp = _mm256_extract_epi64(a, 3);
    s << (temp >> 32) << " " << (temp & 0xFFFFFFFF) << ' ';
    temp = _mm256_extract_epi64(a, 2);
    s << (temp >> 32) << " " << (temp & 0xFFFFFFFF) << ' ';
    temp = _mm256_extract_epi64(a, 1);
    s << (temp >> 32) << " " << (temp & 0xFFFFFFFF) << ' ';
    temp = _mm256_extract_epi64(a, 0);
    s << (temp >> 32) << " " << (temp & 0xFFFFFFFF) << ' ';
    return s;
}

void printvadd(__m256i a, __m256i b) {
   __m256i c = _mm256_add_epi32(a, b);
   cout << c;
}   

void printvadd_andload(const uint32_t* ap, const uint32_t* bp) {
    __m256i a = _mm256_load_si256((__m256i const*)ap);
    __m256i b = _mm256_load_si256((__m256i const*)bp);
   __m256i c = _mm256_add_epi32(a, b);
   cout << c;
}   
int main();
