#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace chrono;

inline void benchmark(const char* name, void (*func)(void*, unsigned int), void* ptr, const unsigned int n) {
	const auto t0 = high_resolution_clock::now();
	func(ptr, n);
	cout << name << " elapsed time: " << setprecision(3) << defaultfloat
		 << duration_cast<nanoseconds>(high_resolution_clock::now() - t0).count() * 1e-9 << " sec" << endl;
}
