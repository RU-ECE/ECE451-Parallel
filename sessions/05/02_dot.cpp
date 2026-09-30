#include <omp.h>
#include <iostream>
using namespace std;
double dot(const double a[], const double b[],
     int n) {
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

int main() {
    const int n = 100'000'000;
    double* a = new double[n];
    double* b = new double[n];
    for (int i = 0; i < n; i++)
      a[i] = b[i] = i;
      cout << omp_get_max_threads() << endl;
    auto t0 = omp_get_wtime();
    cout << dot(a, b, n);
    auto t1 = omp_get_wtime();
    cout << "Time: " << t1 - t0 << endl;
    delete [] a;
    delete [] b;
    return 0;
}