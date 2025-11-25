#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

using namespace std;

// example n = 1000000001
// O(sqrt n)  omega(1)
inline bool isPrime(const uint64_t n) {
	const uint64_t lim = sqrt(n);
	for (uint64_t i = 2; i <= lim; i++)
		if (n % i == 0) // if n is evenly divisible by i
			return false;
	return true;
}
// do the counting in a register, do not write to memory
void countPrimesMultithreaded(uint64_t a, const uint64_t b, uint64_t* pcount) {
	uint64_t count = (a == 2 ? 1 : 0);
	a |= 1; // 10101010101010101010101010100001
	// THE ABOVE GUARANTEES THAT a IS odd
	for (int i = a; i <= b; i += 2) // O(n)
		if (isPrime(i)) // O(sqrt(n))
			count++;
	*pcount = count;
}

int main(int argc, char* argv[]) {
	uint64_t n = atol(argv[1]);
	uint64_t count1 = 0, count2 = 0;
	thread t1(countPrimesMultithreaded, 2, n / 2, &count1);
	thread t2(countPrimesMultithreaded, n / 2 + 1, n, &count2);
	t1.join();
	t2.join();
	const uint64_t count = count1 + count2;
	cout << count << endl;
}
