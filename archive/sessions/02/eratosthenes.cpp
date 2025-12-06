/*
 * 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
 * 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,
 * 1  1  0     0     0      0       0       0       0       0       0       0       0       0       0       0       0
 * 			0        0           0           0           0           0           0           0           0
 * 						 0                   0                   0                   0
 */

unsigned long eratosthenes(const unsigned long n) {
	auto count = 0UL;
	const auto isPrime = new bool[n + 1];
	// first assume all numbers are prime
	for (auto i = 2UL; i <= n; i++)
		isPrime[i] = true;
	for (auto i = 2UL; i <= n; i++) {
		if (isPrime[i]) {
			count++;
			for (auto j = 2UL * i; j <= n; j += i)
				isPrime[j] = false;
		}
	}
	delete[] isPrime;
	return count;
}

/*
	2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
	1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,
	1  1  0     0     0      0       0       0       0       0       0       0       0       0       0       0       0
						 0                       0                       0

	O(n log log n)
*/

unsigned long improved_eratosthenes(const unsigned long n) {
	auto count = 0UL;
	const auto isPrime = new bool[n + 1];
	// first assume all numbers are prime
	for (auto i = 2UL; i <= n; i++)
		isPrime[i] = true;
	for (auto i = 2UL; i <= n; i++) {
		if (isPrime[i]) {
			count++;
			for (auto j = i * i; j <= n; j += 2 * i)
				isPrime[j] = false;
		}
	}
	delete[] isPrime;
	return count;
}

bool isPrime(const unsigned long* primes, const unsigned long i) {
	return primes[i / 64] & 1ULL << (i & 63);
	// 8421
	// 1010
	//  101
	// 10101001010101010110X1011001010101010011111111111111111111
	// 0000000000000000000000000000000000000000000000000000000001
	// i % 64 == i & 63
	//              1000000
	//                 111111
}

/*
	suggested optimizations
	don't store even numbers

	use prime number wheel
	2, 3 = 2*3 = 6

	6,  7,  8,  9, 10, 11,
	12, 13, 14, 15, 16, 17,
	18, 19, 20, 21, 22, 23,
	0    1  2,  3,   4,  5
		 X               X
	mod(6)


	 2,3, 5, 7 = 210
	3, 5, 7 = 105

	105 is not an even multiple of 64
	lcm(105, 64) = 105*64 = 6720

https://en.wikipedia.org/wiki/Wheel_factorization

	segmented eratosthenes


	1st pass is the worst: n/2
	2nd pass: n/3
	3rd pass: n/5
	4th pass: n/7
*/

unsigned long improved_bitpacked_eratosthenes(const unsigned long n) {
	auto count = 0UL;
	const auto SIZE = (n + 63 + 1) / 64UL;
	auto primes = new unsigned long[SIZE];
	// first assume all numbers are prime
	for (auto i = 2UL; i <= SIZE; i++)
		primes[i] = 0xFFFFFFFFFFFFFFFFLL; // MAX(uint64_t,uint64_t)

	for (auto i = 2UL; i <= n; i++) {
		if (isPrime(primes, i)) {
			count++;
			for (auto j = i * i; j <= n; j += 2 * i)
				clearPrime(primes, j);
		}
	}
	delete[] primes;
	return count;
}

/*
 * Hard to parallelize first step, because we need to compute numbers up to sqrt(n) in order to do eratosthenes up to n
 *
 * Stanley: can we come up with some kind of algorithm to split the first kernel
 */
