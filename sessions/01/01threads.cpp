#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using namespace std;

// 28 = 1 ,2 , 4    |   7,    14,  28

// O(sqrt(n))  omega(1)    is n prime? n=1001    n%2, n%3,  .. n%33 n%500
// n=28    1, 2,4    |   7,    14,  28
bool isPrime(const unsigned long n) {
	for (auto i = 2; i <= sqrt(n); i++)
		if (n % i == 0)
			return false;
	return true;
}

[[noreturn]] void f() {
	for (;;) { // infinite loop
		cout << "hello" << flush;
		usleep(100'000);
	}
}

[[noreturn]] void g() {
	while (true) {
		// int* p = new int[1024L * 1024 * 1024 * 16];
		cout << "bye" << flush;
		usleep(200'000);
	}
}

int main() {
	try {
		thread t2(g);
		thread t1(f);
		t1.join();
		t2.join();
	} catch (exception& e) {
		cout << e.what() << endl;
	}
}
