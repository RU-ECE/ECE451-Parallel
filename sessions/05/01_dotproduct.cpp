#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std;
double dot_product(const double a[], const double b[], int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}


double dot_product(const __m256d a[], const __m256d b[], int n) {
    __m256d sum = _mm256_set1_pd(0);
    for (int i = 0; i < n/4; i++) {
    __m256d temp = _mm256_mul_pd (a[i], b[i]);
     sum =  _mm256_add_pd (sum, temp);
    }
    // sum = (s1, s2, s3, s4)
    // sum = (__, __, s1+s3, s2+s4)
    // sum = (__, ___, ____, s1+s2+s3+s4)
    // return low part of sum
    // perform a horizontal add to get a single sum
    double s = sum[0] + sum[1] + sum[2] + sum[3];
    return s;
}

int main() {
    const int n = 1024*1024;
    double* x = (double*)aligned_alloc(sizeof(__m256d), n*sizeof(double));
    double* y = (double*)aligned_alloc(sizeof(__m256d), n*sizeof(double));
    for (int i = 0; i < n; i++) {
       x[i] = i;
       y[i] = i;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    double s1 = dot_product(x, y, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    cout << "s1=" << s1 << "\telapsed: " << elapsed.count() << "usec\n";

    t0 = std::chrono::high_resolution_clock::now();
    double s2 = dot_product((const __m256d*)x, (const __m256d*)y, n);
    t1 = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    cout << "s2=" << s2 << "\telapsed: " << elapsed.count() << "usec\n";
}