//openmpsum.cpp

#include <omp.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;
int main() {
    cout << "omp_get_num_procs():   " << omp_get_num_procs() << endl;
    cout << "omp_get_num_threads(): " <<  omp_get_num_threads() << endl;
    cout << "omp_get_thread_num():  " << omp_get_thread_num() << endl;

    // by default, allocate # threads "optimal" for your system
    //    omp_set_num_threads(atoi(getenv("OMP_NUM_THREADS")));
    //    omp_set_num_threads(4); // set the number of threads manually (bad idea)

	constexpr auto n = 1'000'000'000;
	const auto a = new uint32_t[n];
    for (auto i = 0; i < n; i++)
      a[i] = i;
    // known sum

	const high_resolution_clock::time_point t0 = high_resolution_clock::now();

    uint64_t sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (auto i =0; i < n; i++)
        sum += a[i];
	const high_resolution_clock::time_point t1 = high_resolution_clock::now();
    cout << "Elapsed time: " << duration_cast<duration<double>>(t1 - t0).count() << endl;

    cout << "sum=   " << sum << endl;  
    delete [] a;
}