#include <cmath>
#include <cstdint>
#include <iostream>

using namespace std;

/*
 * hw0: write a multi-threaded prime number counter
 */

bool isPrime(const uint64_t n) {
	for (uint64_t i = 2; i <= sqrt(n); i++)
		if (n % i == 0)
			return false;
	return true;
}

/**
	@return the number of primes up to n
*/
uint64_t count_primes(const uint64_t n) {
	uint64_t count = 0;
	for (uint64_t i = 2; i <= n; i++)
		if (isPrime(i))
			count += 1;
	return count;
}

int main() {
	cout << count_primes(1001) << endl; // should be 168
}
