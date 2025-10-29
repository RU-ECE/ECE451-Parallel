#include <iostream>
#include "benchmark.hpp"
#include <vector>
#include <omp.h>

using namespace std;

/*
    For simplicity, we will represent a matrix as a 1D vector of float with n*n elements.
    We will use the following functions:
    - void multiply_matrices(float* a, float* b, float* c, int n)
    - void multiply_matrices_omp(float* a, float* b, float* c, int n)
    - void multiply_matrices_omp_simd(float* a, float* b, float* c, int n)
    - void multiply_matrices_omp_simd_unroll(float* a, float* b, float* c, int n)
    - void multiply_matrices_omp_simd_unroll_2(float* a, float* b, float* c, int n)
    - void multiply_matrices_omp_simd_unroll_3(float* a, float* b, float* c, int n)
    - use tiling to improve cache locality
      void multiply_matrices_tiling(float* a, float* b, float* c, int n)
    - use blocking to improve cache locality
      void multiply_matrices_blocking(float* a, float* b, float* c, int n)
    - use transposition to improve sequential memory performance
      void multiply_matrices_transposition(float* a, float* b, float* c, int n)
*/


void multiply_matrices(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

void multiply_matrices_omp(float* a, float* b, float* c, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}


void multiply_matrices_omp_simd(float* a, float* b, float* c, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j += 8) {
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < n; k++) {
                __m256 a_vec = _mm256_set1_ps(a[i * n + k]);
                __m256 b_vec = _mm256_loadu_ps(&b[k * n + j]);
                sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
            }
            _mm256_storeu_ps(&c[i * n + j], sum);
        }
    }
}

void multiply_matrices_omp_simd_unroll(float* a, float* b, float* c, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j += 16) {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            for (int k = 0; k < n; k++) {
                __m256 a_vec = _mm256_set1_ps(a[i * n + k]);
                sum0 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&b[k * n + j]), sum0);
                sum1 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&b[k * n + j + 8]), sum1);
            }
            _mm256_storeu_ps(&c[i * n + j], sum0);
            _mm256_storeu_ps(&c[i * n + j + 8], sum1);
        }
    }
}

void multiply_matrices_omp_simd_unroll_2(float* a, float* b, float* c, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i += 2) {
        for (int j = 0; j < n; j += 2) {
            float sum[4] = {0, 0, 0, 0};
            for (int k = 0; k < n; k++) {
                float a0 = a[(i) * n + k];
                float a1 = a[(i+1) * n + k];
                sum[0] += a0 * b[k * n + j];
                sum[1] += a0 * b[k * n + j + 1];
                sum[2] += a1 * b[k * n + j];
                sum[3] += a1 * b[k * n + j + 1];
            }
            c[(i) * n + j] = sum[0];
            c[(i) * n + j + 1] = sum[1];
            c[(i+1) * n + j] = sum[2];
            c[(i+1) * n + j + 1] = sum[3];
        }
    }
}

void multiply_matrices_omp_simd_unroll_3(float* a, float* b, float* c, int n) {
    const int tile = 32;
    #pragma omp parallel for
    for (int i = 0; i < n; i += tile) {
        for (int j = 0; j < n; j += tile) {
            for (int ii = 0; ii < tile && i+ii < n; ii++) {
                for (int jj = 0; jj < tile && j+jj < n; jj++) {
                    float sum = 0;
                    for (int k = 0; k < n; k++) {
                        sum += a[(i+ii) * n + k] * b[k * n + j + jj];
                    }
                    c[(i+ii) * n + j + jj] = sum;
                }
            }
        }
    }
}

void multiply_matrices_tiling(float* a, float* b, float* c, int n) {
    const int tile = 64;
    #pragma omp parallel for
    for (int i = 0; i < n; i += tile) {
        for (int j = 0; j < n; j += tile) {
            for (int k = 0; k < n; k += tile) {
                for (int ii = 0; ii < tile && i+ii < n; ii++) {
                    for (int jj = 0; jj < tile && j+jj < n; jj++) {
                        float sum = 0;
                        for (int kk = 0; kk < tile && k+kk < n; kk++) {
                            sum += a[(i+ii) * n + k + kk] * b[(k+kk) * n + j + jj];
                        }
                        c[(i+ii) * n + j + jj] += sum;
                    }
                }
            }
        }
    }
}



int main() {
    int n = 1000;
    vector<float> a(n*n);
    vector<float> b(n*n);
    vector<float> c(n*n);
    for (int i = 0; i < n*n; i++) {
        a[i] = 1;
        b[i] = 1;
        c[i] = 0;
    }
    benchmark(multiply_matrices, &a[0], &b[0], &c[0], n);
    for (int i = 0; i < n*n; i++) c[i] = 0;
    benchmark(multiply_matrices_omp, &a[0], &b[0], &c[0], n);
    for (int i = 0; i < n*n; i++) c[i] = 0;
    benchmark(multiply_matrices_omp_simd, &a[0], &b[0], &c[0], n);
    for (int i = 0; i < n*n; i++) c[i] = 0;
    benchmark(multiply_matrices_omp_simd_unroll, &a[0], &b[0], &c[0], n);
    for (int i = 0; i < n*n; i++) c[i] = 0;
    benchmark(multiply_matrices_omp_simd_unroll_2, &a[0], &b[0], &c[0], n);
    for (int i = 0; i < n*n; i++) c[i] = 0;
    benchmark(multiply_matrices_omp_simd_unroll_3, &a[0], &b[0], &c[0], n);
    for (int i = 0; i < n*n; i++) c[i] = 0;
    benchmark(multiply_matrices_tiling, &a[0], &b[0], &c[0], n);
    return 0;
}

