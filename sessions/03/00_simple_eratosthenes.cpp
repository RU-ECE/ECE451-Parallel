#include <cmath>
#include <iostream>

using namespace std;

// O(log log n)
unsigned long eratosthenes(bool primes[], const unsigned long n) {
	auto count = 0UL;
	for (auto i = 2UL; i <= n; i++)
		primes[i] = true;

	for (auto i = 2UL; i <= n; i++) {
		if (primes[i]) {
			count++;
			for (unsigned long j = 2 * i; j <= n; j += i)
				primes[j] = false;
		}
	}
	return count;
}

int main(const int argc, char* argv[]) {
	const auto n = argc > 1 ? strtol(argv[1], nullptr, 10) : 1000;
	const auto primes = new bool[n + 1];
	cout << eratosthenes(primes, n) << endl;
	delete[] primes;
}
