#include <iostream>
#include <thread>

using namespace std;

auto count_bananas = 0;

void grow_bananas(const int n) {
	for (auto i = 0; i < n; i++)
		count_bananas++; // read count_bananas, increment, write back
}

void eat_bananas(const int n) {
	for (auto i = 0; i < n; i++)
		count_bananas--; // read count_bananas, decrement, write back
}

constexpr auto n = 10;

void threading1() {
	thread t1(grow_bananas, n);
	thread t2(eat_bananas, n);
	cout << count_bananas << endl;
}

void threading2() {
	thread t1(grow_bananas, n);
	t1.join();
	thread t2(eat_bananas, n);
	t2.join();
	cout << count_bananas << ' ';
}

void threading3() {
	thread t1(grow_bananas, n);
	thread t2(eat_bananas, n);
	t1.join();
	t2.join();
	cout << count_bananas << endl;
}

int main() {
	// threading1();
	// threading2();
	threading3();
}
