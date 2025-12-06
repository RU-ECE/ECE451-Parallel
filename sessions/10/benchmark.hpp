#pragma once
#include <chrono>
#include <iostream>

using namespace std;
using namespace chrono;

inline void benchmark(void (*func)(const float*, const float*, float*, int), const float* a, const float* b, float* c,
					  const int n) {
	const auto start = high_resolution_clock::now();
	func(a, b, c, n);
	cout << "Time taken by function: " << duration_cast<microseconds>(high_resolution_clock::now() - start).count()
		 << " microseconds" << endl;
}
