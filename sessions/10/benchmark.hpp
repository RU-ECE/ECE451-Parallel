#pragma once
#include <chrono>
#include <iostream>

using namespace std;
using namespace chrono;

inline void benchmark(void (*func)(const float*, const float*, float*, int), const float* a, const float* b, float* c,
					  const int n) {
	const auto start = high_resolution_clock::now();
	func(a, b, c, n);
	const auto end = high_resolution_clock::now();
	const auto duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
}
