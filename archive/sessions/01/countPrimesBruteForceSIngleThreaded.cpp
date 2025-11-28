#include <cmath>
#include <cstdint>
#include <iostream>

using namespace std;

uint64_t count = 0;

// return true if n is prime O(sqrt n)
bool isPrime(const uint64_t n) {
	const uint64_t lim = sqrt(n);
	for (uint64_t i = 2; i <= lim; i++)
		if (n % i == 0) // if n is evenly divisible by i
			return false;
	return true;
}

// count all primes from 2 to n
uint64_t countPrimes(const uint64_t n) {
	uint64_t count = 0;
	for (uint64_t i = 2; i <= n; i++)
		if (isPrime(i))
			count++;
	return count;
}

int main(int argc, char* argv[]) {
	const __uint64_t n = atoi(argv[1]);
	cout << countPrimes(n) << endl;
	return 0;
}
