#include <iostream>
#include <thread>

using namespace std;

/*
 * race condition
 */

int g(int x);
int f(int x);

auto count = 0; // initialize global variable to zero (good style)
constexpr auto n = 100'000'000;

void increment() {
	for (auto i = 0; i < n; i++)
		count = f(count);
}

void decrement() {
	for (auto i = 0; i < n; i++)
		count = g(count);
}

int main() {
	thread t1(increment);
	thread t2(decrement);
	t1.join();
	t2.join();

	cout << count << endl;
	return 0;
}
