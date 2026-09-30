#include <omp.h>

#include <iostream>
#include <cmath>

using namespace std;

int main() {
    // I have 8 cores (hyperthreading) each of which has
    //   registers
    //   each pair of cores shares SOME execution units
    //  each core can add

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cout << "Hello from thread " << tid << endl;
    }
    return 0;
}