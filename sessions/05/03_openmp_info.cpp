#include <omp.h>
#include <iostream>

int main() {
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int max_threads = omp_get_max_threads();
        int thread_limit = omp_get_thread_limit();
        std::cout << "thread: " << omp_get_thread_num() << " of " << omp_get_num_threads() 
                  << " num_threads: " << num_threads << " max_threads: " << max_threads << " thread_limit: " << thread_limit << "\n";
    }
    return 0;
}