#include <iostream>
#include <cstdint>
#include <immintrin.h>
#include <iomanip>
using namespace std;

double sum(uint64_t a, uint64_t b) {
  double sum = 0;
  for (uint64_t i = a; i <= b; i++) {
    sum += 1.0/(i*i);
  }
  return sum;
}

double sum_avx(uint64_t a, uint64_t b) {
  __m256d sum = _mm256_setzero_pd();
  double num[4] = {1, 1, 1, 1};
  double terms[4] = {1, 2, 3, 4};
  __m256d num_vec = _mm256_loadu_pd(num);
  __m256d four = _mm256_set1_pd(4);
  __m256d term = _mm256_loadu_pd((double*)&terms);
  for (uint64_t i = a; i <= b; i += 4) {
    __m256d square = _mm256_mul_pd(term, term);
    __m256d t = _mm256_div_pd(num_vec, square);
    sum = _mm256_add_pd(sum, t);
    term = _mm256_add_pd(term, four);
  }
  double result[4]; // this code is AI slop. There is way better way to do this
  _mm256_storeu_pd(result, sum);
  double total = 0;
  for (int i = 0; i < 4; i++) {
    total += result[i];
  }
  return total;
}

int main() {
  const uint64_t n = 800'000;
  cout << setprecision(15) << sum(1, n) << endl;
  cout << setprecision(15) << sum_avx(1, n) << endl;
}