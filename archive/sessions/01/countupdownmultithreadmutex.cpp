#include <iostream>
#include <mutex>
#include <thread>

using namespace std;

mutex m;

/*
 * race condition
 */

int g(int x);
int f(int x);

auto count = 0; // initialize global variable to zero (good style)
constexpr auto n = 10'000'000;

void increment() {
	for (auto i = 0; i < n; i++) {
		lock_guard lock(m);
		count = f(count);
	}
}

void decrement() {
	for (auto i = 0; i < n; i++) {
		lock_guard lock(m);
		count = g(count);
	}
}

int main() {
	thread t1(increment);
	thread t2(decrement);
	t1.join();
	t2.join();
	cout << count << endl;
	return 0;
}
