#include <immintrin.h>

void sort14incpp(int a[]) {
    _m256i a0 = _mm256_loadu_si256((__m256i*)a);
    _m256i a1 = _mm256_loadu_si256((__m256i*)a+8);
    _m256i a2 = _mm256_loadu_si256((__m256i*)a+16);
    _m256i a3 = _mm256_loadu_si256((__m256i*)a+24);
    _m256i a4 = _mm256_loadu_si256((__m256i*)a+32);
    _m256i a5 = _mm256_loadu_si256((__m256i*)a+40);
    _m256i a6 = _mm256_loadu_si256((__m256i*)a+48);
    _m256i a7 = _mm256_loadu_si256((__m256i*)a+56);
    _m256i a8 = _mm256_loadu_si256((__m256i*)a+64);
    _m256i a9 = _mm256_loadu_si256((__m256i*)a+72);
    _m256i a10 = _mm256_loadu_si256((__m256i*)a+80);

}



extern "C" void sortoptimal14(int a[]);
void sort14(int a[], int n) {
    for (int i = 0; i < n; i += 14*8)
    sortoptimal14(a+i);
}

int main() {
    int a[128];
    for (int i = 0; i < 128; i++)
      a[i] = 128-i;
    sortoptimal14(a);
    return 0;
}