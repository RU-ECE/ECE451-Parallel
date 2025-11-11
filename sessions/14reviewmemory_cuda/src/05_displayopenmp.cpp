#include <omp.h>
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

int main() {

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int max_threads = omp_get_max_threads();
        int thread_limit = omp_get_thread_limit();
        cout << tid << " " << nthreads << " " << max_threads << " " << thread_limit << endl;
    }
    return 0;
}