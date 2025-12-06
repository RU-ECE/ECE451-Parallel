#include <cstdio>

// 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
// 0  0  1  0  1  0  1  1  1   0   1   0   1   1   1   0                               1
// i=97
// O(log log n)

void eratosthenes(bool primes[], const unsigned long n) {
	auto count = 0UL;
	for (auto i = 2UL; i <= n; i++)
		primes[i] = true;

	for (auto i = 2UL; i <= n; i++) {
		if (primes[i]) {
			count++;
			for (auto j = i * i; j <= n; j += 2 * i)
				primes[j] = false;
		}
	}
	// *pcount = count;
}

int main() {
	constexpr auto n = 100UL;
	bool primes[n + 1];
	eratosthenes(primes, n);
	for (auto i = 0UL; i <= n; i++)
		if (primes[i])
			printf("%lu ", i);
}
