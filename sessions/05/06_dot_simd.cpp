#include <chrono>
#include <iostream>
using namespace std;

// single thread; vector lanes each keep a partial sum, combined at the end
// build: g++ -O2 -march=native -fopenmp-simd 06_dot_simd.cpp
// (-fopenmp-simd honors only simd pragmas, no thread runtime)
double dot(const double a[], const double b[], int n) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

int main() {
    const int n = 100'000'000;
    double* a = new double[n];
    double* b = new double[n];
    for (int i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }
    auto t0 = chrono::steady_clock::now();
    double result = dot(a, b, n);
    auto t1 = chrono::steady_clock::now();
    cout << "dot=" << result << " expected=" << 2.0 * n
         << " time=" << chrono::duration<double>(t1 - t0).count() << '\n';
    delete[] a;
    delete[] b;
    return 0;
}
