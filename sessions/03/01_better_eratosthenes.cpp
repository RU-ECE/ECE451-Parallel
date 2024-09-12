#include <iostream>
#include <cmath>
using namespace std;

//O(log log n)
uint64_t eratosthenes(bool primes[], uint64_t n) {
	uint64_t count = 1; //special case, 2 is prime
	for (uint64_t i = 3; i <= n; i += 2) {
		primes[i] = true; // only write odd ones
	}

	for (uint64_t i = 3; i <= n; i += 2) {
		if (primes[i]) {
			count++;
			for (uint64_t j = i * i; j <= n; j += 2*i) {
				primes[j] = false;
			}
		}
	}
	return count;
}

int main(int argc, char* argv[]) {
	uint64_t n = argc > 1 ? atol(argv[1]) : 1000;
	bool primes[n+1];
	cout << eratosthenes(primes, n) << '\n';
}
