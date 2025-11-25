#include <iostream>
#include <omp.h>
int main() {
	omp_set_num_threads(4);
#pragma omp parallel
	{
		const int threadid = omp_get_thread_num();
#pragma omp critical
		{
			std::cout << "Thread id : " << threadid << std::endl;
		}
	}
	return 0;
}
