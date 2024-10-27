#include <omp.h>
#include <iostream>

/*
    Note: when we compiled with g++ -O3 -fopenmp -o sum_simd sum_simd.cpp this only did SSE
    to do AVX:
        g++ -O3 -mavx2 -fopenmp -o sum_simd sum_simd.cpp
*/

int main() {
    const int n = 16;
    float* a = new float[n];
    float* b = new float[n];
    float* c = new float[n];
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    #pragma omp parallel for simd
    for(int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }

    for(int i = 0; i < n; i++) {
        std::cout << c[i] << ' ';
    }
    std::cout << '\n';
    delete [] a;
    delete [] b;
    delete [] c;
}