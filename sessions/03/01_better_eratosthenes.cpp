#include <cmath>
#include <cstdint>
#include <iostream>

using namespace std;

// O(log log n)
uint64_t eratosthenes(bool primes[], const uint64_t n) {
	uint64_t count = 1; // special case, 2 is prime
	for (uint64_t i = 3; i <= n; i += 2)
		primes[i] = true; // only write odd ones
	const uint64_t lim = sqrt(n);
	for (uint64_t i = 3; i <= lim; i += 2) {
		if (primes[i]) {
			count++;
			for (uint64_t j = i * i; j <= n; j += 2 * i)
				primes[j] = false;
		}
	}
	// if (lim% 2 != 0) {
	// 	lim += 2;
	// } else {
	// 	lim += 1;
	// }
	// (lim + 1) | 1 means round up to next odd number
	for (uint64_t i = lim + 1 | 1; i <= n; i += 2)
		if (primes[i])
			count++;
	return count;
}

int main(const int argc, char* argv[]) {
	const uint64_t n = argc > 1 ? atol(argv[1]) : 1000;
	// all modern OS will crash if you allocate more than 4MB on stack
	int x;
	int y[1024];
	int z[1024 * 1024]; // BOOM
	const auto primes = new bool[n + 1];
	cout << eratosthenes(primes, n) << endl;
	delete[] primes;
}
