#include <omp.h>
#include <chrono>
#include <iostream>

//g++ -O3 -fopenmp -mavx2 -mfma -S -masm=intel omp_demo.cpp -o omp_demo.s

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <vector>
using namespace std;

constexpr uint64_t N = 16'000'000;
constexpr int REPEATS = 8;

void initialize(vector<double>& a, vector<double>& b) {
    #pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < N; ++i) {
        a[i] = 1.0 + static_cast<double>(i % 1000) * 0.001;
        b[i] = 2.0 - static_cast<double>(i % 500) * 0.002;
    }
}

void compute_serial(const double* a, const double* b, double* c) {
    for (uint64_t i = 0; i < N; ++i)
        c[i] = a[i] * 1.25 + b[i] * 0.75;
}

void compute_parallel(const double* a, const double* b, double* c) {
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < N; ++i)
        c[i] = a[i] * 1.25 + b[i] * 0.75;
}

void compute_dynamic(const double* a, const double* b, double* c) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (uint64_t i = 0; i < N; ++i)
        c[i] = a[i] * 1.25 + b[i] * 0.75;
}

void compute_simd(const double* a, const double* b, double* c) {
    #pragma omp simd
    for (uint64_t i = 0; i < N; ++i)
        c[i] = a[i] * 1.25 + b[i] * 0.75;
}

void compute_parallel_simd(const double* a, const double* b, double* c) {
    #pragma omp parallel for simd schedule(static)
    for (uint64_t i = 0; i < N; ++i)
        c[i] = a[i] * 1.25 + b[i] * 0.75;
}

template<typename Function>
double benchmark(Function function, const double a[], const double b[], double* c) {
    function(a, b, c);  // Warm-up

    const auto start = chrono::steady_clock::now();

    for (int i = 0; i < REPEATS; ++i)
        function(a, b, c);

    const auto stop = chrono::steady_clock::now();

    return chrono::duration<double>(stop - start).count() / REPEATS;
}

int main(int argc, char** argv) {
    int threads = omp_get_max_threads();

    if (argc > 1)
        threads = max(1, atoi(argv[1]));

    omp_set_num_threads(threads);

    vector<double> a(N), b(N), c(N);

    initialize(a, b);

    const double serial = benchmark(compute_serial, a.data(), b.data(), c.data());
    const double parallel = benchmark(compute_parallel, a.data(), b.data(), c.data());
    const double dynamic = benchmark(compute_dynamic, a.data(), b.data(), c.data());
    const double simd = benchmark(compute_simd, a.data(), b.data(), c.data());
    const double parallel_simd = benchmark(compute_parallel_simd, a.data(), b.data(), c.data());

    // Prevent the final result from being entirely unused.
    double checksum = 0.0;
    for (uint64_t i = 0; i < N; i += 1024)
        checksum += c[i];

    printf("Elements: %'llu\n", static_cast<unsigned long long>(N));
    printf("Threads:  %d\n\n", threads);
    printf("%-22s %12s %12s\n", "Method", "Time (s)", "Speedup");
    printf("%-22s %12.6f %12.2fx\n", "Serial", serial, 1.0);
    printf("%-22s %12.6f %12.2fx\n", "Parallel static", parallel, serial / parallel);
    printf("%-22s %12.6f %12.2fx\n", "Dynamic, chunk 1", dynamic, serial / dynamic);
    printf("%-22s %12.6f %12.2fx\n", "SIMD only", simd, serial / simd);
    printf("%-22s %12.6f %12.2fx\n", "Parallel + SIMD", parallel_simd, serial / parallel_simd);
    printf("\nChecksum: %.9f\n", checksum);

    return 0;
}
