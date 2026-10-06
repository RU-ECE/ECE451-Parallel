#include <omp.h>
#include <iostream>
using namespace std;

double dot(const double a[], const double b[], int n) {
    double sum = 0.0;
    // each thread gets a private sum initialized to 0, combined with + at the end
    #pragma omp parallel for reduction(+:sum)
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
    double t0 = omp_get_wtime();
    double result = dot(a, b, n);
    double t1 = omp_get_wtime();
    cout << "dot=" << result << " expected=" << 2.0 * n
         << " time=" << t1 - t0 << " threads=" << omp_get_max_threads() << '\n';
    delete[] a;
    delete[] b;
    return 0;
}
