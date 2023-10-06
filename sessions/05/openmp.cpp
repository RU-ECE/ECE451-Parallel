#include <iostream>
#include <omp.h>

using namespace std;

constexpr int n = 400'000'000;
double *a;
double *b;

void f();
int main() {
    a = new double[n];
    b = new double[n];
    f();

    delete [] a;
    delete [] b;
}

void f() {
    clock_t t0 = clock();
    #pragma omp parallel for
    for (int i=1; i<n; i++) /* i is private by default */
        b[i] = (a[i] + a[i-1]) / 2.0;
    clock_t t1 = clock();
    cout << "Elapsed time: " << (double)(t1 - t0) / CLOCKS_PER_SEC << endl;
// at this point, the loop is done

}

