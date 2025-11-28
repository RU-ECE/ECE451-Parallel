#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace chrono;

inline void benchmark(const char* name, void (*func)(void*, unsigned int), void* ptr, const unsigned int n) {
	const auto t0 = high_resolution_clock::now();
	func(ptr, n);
	const auto t1 = high_resolution_clock::now();
	const auto duration = duration_cast<nanoseconds>(t1 - t0);
	const double seconds = duration.count() * 1e-9;
	cout << name << " elapsed time: " << setprecision(3) << defaultfloat << seconds << " sec" << endl;
}
