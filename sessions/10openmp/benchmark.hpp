#pragma once
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

void benchmark(void (*func)(float*, float*, float*, int), float* a, float* b, float* c, int n) {
    auto start = high_resolution_clock::now();
    func(a, b, c, n);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
}