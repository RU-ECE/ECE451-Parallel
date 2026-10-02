#include <omp.h>
#include <iostream>
using namespace std;

// 9   7   5   1  6   8   3
// -2  -2  -4  5  2   -5  ???


void diff(double x[], int n) {
    for (int i = 0; i < n-1; i++)
       x[i] = x[i+1] - x[i];
}

// 9   7   5   1  6   8   3
// 9   2   2   4  -5  -2  5
void diff2(double x[], int n) {
    for (int i = n-1; i > 0; i--)
       x[i] = x[i-1] - x[i];
}


// now let's use OpenMP
void mpdiff(double x[], int n) {
    #pragma omp parallel 
    {
        int threadid = omp_get_thread_num(); // who am i?
        int numthreads = omp_get_num_threads();
        int chunksize = (n+numthreads-1) / numthreads ; // (30+7)/8= 4
        int start = threadid * chunksize;
        int end = start + chunksize;
        double temp = x[end];
        // wait for all threads to complete
        
        for (int i = start; i < end-1; i++)
           x[i] = x[i+1] - x[i];
        x[end-1] = temp - x[end-1];
    }
}


int main() {
    #pragma omp parallel for
    for (int i = 0; i < 30; i++) {
        string msg = "thread: " + to_string(omp_get_thread_num()) + " i: " + to_string(i) + "\n";
        std::cout << msg;
    }

    // this is not generally a good idea if we are dealing with memory
    // CUDA can do this through very clever memory management but OpenMP cannot
    // BLOCKS is often the best
    cout << "\n\n Now scheduling 1 at a time statically\n\n";
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < 30; i++) {
        string msg = "thread: " + to_string(omp_get_thread_num()) + " i: " + to_string(i) + "\n";
        std::cout << msg;
    }



    return 0;
}