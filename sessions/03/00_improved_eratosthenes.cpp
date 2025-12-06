/*
n      n sqrt(n)        n log n
1      1                1
10     30               33
100    1000             700
10^6   10^9             2*1076
10^9   3.3*10^12
*/

// original eratosthenes
// 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
// 1  1  1  1  1  1  1  1  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1                            1
//       0     0     0     0       0       0       0       0       0       0       0
//             0        0          0           0           0           0           0
//

// improved eratosthenes
// 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
// 1  1  1  1  1  1  1  1  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1                            1
//       0     0     0     0       0       0       0       0       0       0       0
//                      0                      0                       0           0
// i=97*97

// O(n log log n)

#include <cmath>
#include <iostream>

void improved_eratosthenes(bool primes[], const unsigned long n) {
	auto count = 1UL;
	for (auto i = 3UL; i <= n; i += 2)
		primes[i] = true;

	for (auto i = 3UL; i <= static_cast<unsigned long>(sqrt(static_cast<double>(n))); i += 2) {
		if (primes[i]) {
			count++;
			for (auto j = i * i; j <= n; j += 2 * i)
				primes[j] = false;
		}
	}
	for (auto i = static_cast<unsigned long>(sqrt(static_cast<double>(n))) + 1; i <= n; i++)
		if (primes[i])
			count++;
	//    *pcount = count;
}

int main() {
	constexpr auto n = 100UL;
	bool primes[n + 1];
	improved_eratosthenes(primes, n);
	for (auto i = 0UL; i <= n; i++)
		if (primes[i])
			printf("%lu ", i);
}
