#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

using namespace std;

// example n = 1000000001
// O(sqrt n) omega(1)
inline bool isPrime(const uint64_t n) {
	const uint64_t lim = sqrt(n);
	for (uint64_t i = 3; i <= lim; i += 2)
		if (n % i == 0) // if n is evenly divisible by i
			return false;
	return true;
}

// 3, 5, 7, 9, 11, 13, 15, 17, ... 1001, 1'000'000'000
uint64_t countPrimes(const uint64_t n) { // O(n sqrt(n))
	uint64_t count = 1;
	for (auto i = 3; i <= n; i += 2) // O(n)
		if (isPrime(i)) // O(sqrt(n))
			count++;
	return count;
}
// n = 10^9
// 2-1K, 1K+1-2K, 2K+1-3K,

void countPrimesMultithreaded(uint64_t a, const uint64_t b, uint64_t* pcount) {
	// uint64_t count = 1;
	*pcount = a == 2 ? 1 : 0;
	a |= 1; // 10101010101010101010101010100001
	// THE ABOVE GUARANTEES THAT a IS odd
	// if (a % 2 == 0)
	// a++;
	for (int i = a; i <= b; i += 2) // O(n)
		if (isPrime(i)) // O(sqrt(n))
			(*pcount)++;
}

// do the counting in a register, do not write to memory
void countPrimesMultithreaded2(uint64_t a, const uint64_t b, uint64_t* pcount) {
	uint64_t count = a == 2 ? 1 : 0;
	a |= 1; // 10101010101010101010101010100001
	// THE ABOVE GUARANTEES THAT a IS odd
	for (int i = a; i <= b; i += 2) // O(n)
		if (isPrime(i)) // O(sqrt(n))
			count++;
	*pcount = count;
}

int main(int argc, char* argv[]) {
	uint64_t n = atol(argv[1]);
	uint64_t chunkSize = 1024 * 1024;
	uint64_t current = 2;
	// cout << countPrimes(n) << endl;
	uint64_t count1 = 0, count2 = 0;
	thread t1(countPrimesMultithreaded2, 2, n / 2, &count1);
	thread t2(countPrimesMultithreaded2, n / 2 + 1, n, &count2);
	t1.join();
	t2.join();
	const uint64_t count = count1 + count2;
	cout << count << endl;
}
