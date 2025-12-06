#include <cmath>
#include <iostream>

using namespace std;

auto count = 0UL;

// return true if n is prime O(sqrt n)
bool isPrime(const unsigned long n) {
	const auto lim = static_cast<unsigned long>(sqrt(n));
	for (auto i = 2UL; i <= lim; i++)
		if (n % i == 0) // if n is evenly divisible by i
			return false;
	return true;
}

// count all primes from 2 to n
unsigned long countPrimes(const unsigned long n) {
	auto count = 0;
	for (auto i = 2UL; i <= n; i++)
		if (isPrime(i))
			count++;
	return count;
}

int main(int, char* argv[]) {
	const auto n = strtol(argv[1], nullptr, 10);
	cout << countPrimes(n) << endl;
	return 0;
}
