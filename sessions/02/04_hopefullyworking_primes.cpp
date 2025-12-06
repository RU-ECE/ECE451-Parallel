#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

using namespace std;

// example n = 1000000001
// O(sqrt n) omega(1)
inline bool isPrime(const unsigned long n) {
	for (auto i = 2; i <= sqrt(n); i++)
		if (n % i == 0) // if n is evenly divisible by i
			return false;
	return true;
}
// do the counting in a register, do not write to memory
void countPrimesMultithreaded(unsigned long a, const unsigned long b, unsigned long* pcount) {
	auto count = a == 2 ? 1UL : 0;
	a |= 1; // 10101010101010101010101010100001
	// THE ABOVE GUARANTEES THAT a IS odd
	for (auto i = a; i <= b; i += 2) // O(n)
		if (isPrime(i)) // O(sqrt(n))
			count++;
	*pcount = count;
}

int main(int, char* argv[]) {
	auto n = strtol(argv[1], nullptr, 10);
	auto count1 = 0UL, count2 = 0UL;
	thread t1(countPrimesMultithreaded, 2, n / 2, &count1);
	thread t2(countPrimesMultithreaded, n / 2 + 1, n, &count2);
	t1.join();
	t2.join();
	cout << count1 + count2 << endl;
}
