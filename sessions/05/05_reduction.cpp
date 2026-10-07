#include <omp.h>
#include <iostream>
#include <iomanip>
using namespace std;

double dot(const double a[], const double b[], int n) {
    double sum = 0.0;
    // each thread gets a private sum initialized to 0, combined with + at the end
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

double factorial(int n) {
    double prod = 1;
    for (int i = 1; i <= n; i++) 
      prod *= i;
    return prod;
}


// this isn't worth it because factorial isn't enough work, but shows the idea
double omp_factorial(int n) {
    double prod = 1;
    #pragma omp parallel for reduction(*:prod)
    for (int i = 1; i <= n; i++) 
      prod *= i;
    return prod;
}

// this is a BAD idea
double omp_factorial_noreduction(int n) {
    double prod = 1;
    #pragma omp parallel for
    for (int i = 1; i <= n; i++) 
      prod *= i;
    return prod;
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
    int p = 100;
    t0 = omp_get_wtime();
    double fac = factorial(p);
    t1 = omp_get_wtime();
    cout << setprecision(15);
    cout << "omp_factorial=" << fac << "time = " << (t1-t0) << '\n';


    t0 = omp_get_wtime();
    fac = omp_factorial(p);
    t1 = omp_get_wtime();
    cout << setprecision(15);
    cout << "omp_factorial=" << fac << "time = " << (t1-t0) << '\n';

    t0 = omp_get_wtime();
    fac = omp_factorial_noreduction(p);
    t1 = omp_get_wtime();
    cout << setprecision(15);
    cout << "omp_factorial=" << fac << "time = " << (t1-t0) << '\n';
    return 0;
}
