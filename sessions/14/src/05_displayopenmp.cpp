#include <iostream>
#include <omp.h>

using namespace std;

int main() {
#pragma omp parallel
	{
		cout << omp_get_thread_num() << " " << omp_get_num_threads() << " " << omp_get_max_threads() << " "
			 << omp_get_thread_limit() << endl;
	}
	return 0;
}
