#include <iostream>
#include <thread>
#include <chrono>
using namespace std;

void sum(uint64_t n, uint64_t* p) {
  int count = 0;
  for (uint64_t i = 0; i < n; i++)
    count++; // counts fast, in a register
  *p = count; // writes only once to memory
}

void sumslowly(uint64_t n, uint64_t* p) {
  for (uint64_t i = 0; i < n; i++)
    (*p)++; // reads and writes EVERY TIME
}



void testfastsum(uint64_t n) {
    auto start = std::chrono::steady_clock::now();
    uint64_t counts[2] = {0};
    thread t1(sum, n, &counts[0]);
    thread t2(sum, n, &counts[1]);
    t1.join();
    t2.join();
    uint64_t sum = counts[0] + counts[1];
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);

    cout << sum << " elapsed: " << elapsed.count() << '\n';
}

void testslowsum(uint64_t n) {
    auto start = std::chrono::steady_clock::now();
    uint64_t counts[2] = {0};
    thread t1(sumslowly, n, &counts[0]);
    thread t2(sumslowly, n, &counts[1]);
    t1.join();
    t2.join();
    uint64_t sum = counts[0] + counts[1];
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);

    cout << sum << " elapsed: " << elapsed.count() << '\n';
}

int main() {
    const uint64_t n = 10'000'000;
    testfastsum(n);
    testslowsum(n);
}