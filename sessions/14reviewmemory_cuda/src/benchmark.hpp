#pragma once
#include <chrono>
#include <iostream>
#include <iomanip>

void benchmark(const char* name, void (*func)(void*, unsigned int), void* ptr, unsigned int n) {
    auto t0 = std::chrono::high_resolution_clock::now();
    func(ptr, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
    double seconds = duration.count() * 1e-9;
    std::cout << name << " elapsed time: " << std::setprecision(3) << std::defaultfloat << seconds << " sec" << std::endl;
}
