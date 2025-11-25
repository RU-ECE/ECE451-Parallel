#include <chrono>
#include <iostream>
#include <omp.h>

using namespace std;
using namespace chrono;

int main() {

#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		const int nthreads = omp_get_num_threads();
		const int max_threads = omp_get_max_threads();
		const int thread_limit = omp_get_thread_limit();
		cout << tid << " " << nthreads << " " << max_threads << " " << thread_limit << endl;
	}
	return 0;
}
