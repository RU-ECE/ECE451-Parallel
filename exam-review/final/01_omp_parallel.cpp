#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp parallel
	{
		const int thread_id = omp_get_thread_num();
		printf("Hello from thread %d\n", thread_id);
	}
	return 0;
}
