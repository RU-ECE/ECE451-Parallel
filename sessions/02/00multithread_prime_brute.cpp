#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

using namespace std;

// example n = 1000000001
// O(sqrt n) omega(1)
inline bool isPrime(const unsigned long n) {
	const auto lim = static_cast<unsigned long>(sqrt(n));
	for (auto i = 3UL; i <= lim; i += 2)
		if (n % i == 0) // if n is evenly divisible by i
			return false;
	return true;
}

// 3, 5, 7, 9, 11, 13, 15, 17, ... 1001, 1'000'000'000
unsigned long countPrimes(const unsigned long n) { // O(n sqrt(n))
	auto count = 1UL;
	for (auto i = 3UL; i <= n; i += 2) // O(n)
		if (isPrime(i)) // O(sqrt(n))
			count++;
	return count;
}
// n = 10^9
// 2-1K, 1K+1-2K, 2K+1-3K,

void countPrimesMultithreaded(unsigned long a, const unsigned long b, unsigned long* pcount) {
	// uint64_t count = 1;
	*pcount = a == 2 ? 1 : 0;
	a |= 1; // 10101010101010101010101010100001
	// THE ABOVE GUARANTEES THAT a IS odd
	// if (a % 2 == 0)
	// a++;
	for (auto i = a; i <= b; i += 2) // O(n)
		if (isPrime(i)) // O(sqrt(n))
			(*pcount)++;
}

// do the counting in a register, do not write to memory
void countPrimesMultithreaded2(unsigned long a, const unsigned long b, unsigned long* pcount) {
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
	auto chunkSize = 1024 * 1024;
	auto current = 2;
	// cout << countPrimes(n) << endl;
	auto count1 = 0UL, count2 = 0UL;
	thread t1(countPrimesMultithreaded2, 2, n / 2, &count1);
	thread t2(countPrimesMultithreaded2, n / 2 + 1, n, &count2);
	t1.join();
	t2.join();
	cout << count1 + count2 << endl;
}
