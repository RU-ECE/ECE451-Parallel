#include <cmath>
#include <iostream>

using namespace std;

// O(log log n)
unsigned long eratosthenes(bool primes[], const unsigned long n) {
	auto count = 1UL; // special case, 2 is prime
	for (auto i = 3UL; i <= n; i += 2)
		primes[i] = true; // only write odd ones
	const auto lim = static_cast<unsigned long>(sqrt(n));
	for (auto i = 3UL; i <= lim; i += 2) {
		if (primes[i]) {
			count++;
			for (auto j = i * i; j <= n; j += 2 * i)
				primes[j] = false;
		}
	}
	// if (lim% 2 != 0) {
	// 	lim += 2;
	// } else {
	// 	lim += 1;
	// }
	// (lim + 1) | 1 means round up to next odd number
	for (auto i = lim + 1 | 1; i <= n; i += 2)
		if (primes[i])
			count++;
	return count;
}

int main(const int argc, char* argv[]) {
	const auto n = argc > 1 ? strtol(argv[1], nullptr, 10) : 1000;
	// all modern OS will crash if you allocate more than 4MB on stack
	int x;
	int y[1024];
	int z[1024 * 1024]; // BOOM
	const auto primes = new bool[n + 1];
	cout << eratosthenes(primes, n) << endl;
	delete[] primes;
}
