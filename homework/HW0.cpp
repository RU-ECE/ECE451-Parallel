#include <cmath>
#include <iostream>

using namespace std;

/*
 * hw0: write a multi-threaded prime number counter
 */

bool isPrime(const unsigned long n) {
	for (auto i = 2; i <= sqrt(n); i++)
		if (n % i == 0)
			return false;
	return true;
}

/**
	@return the number of primes up to n
*/
unsigned long count_primes(const unsigned long n) {
	auto count = 0UL;
	for (auto i = 2UL; i <= n; i++)
		if (isPrime(i))
			count += 1;
	return count;
}

int main() {
	cout << count_primes(1001) << endl; // should be 168
}
