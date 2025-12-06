#include <iostream>
#include <omp.h>

using namespace std;

int main() {
	omp_set_num_threads(4);
#pragma omp parallel
	{
		const auto threadid = omp_get_thread_num();
#pragma omp critical
		{
			cout << "Thread id : " << threadid << endl;
		}
	}
	return 0;
}
