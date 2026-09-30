#include <omp.h>

#include <iostream>
#include <cmath>

using namespace std;

int main() {
    // I have 8 cores (hyperthreading) each of which has
    //   registers
    //   each pair of cores shares SOME execution units
    //  each core can add
    const int n = 80000;
    int* a = new int[n];

    // each thread will execute 1/8 of the loop
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = i;
//        #pragma omp critical
        cout << i << ": Hello from thread " << omp_get_thread_num() << endl;
    }
    return 0;
}