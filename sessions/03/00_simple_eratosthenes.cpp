#include <iostream>
#include <cmath>
using namespace std;

//O(log log n)
uint64_t eratosthenes(bool primes[], uint64_t n) {
    uint64_t count = 0;
    for (uint64_t i = 2; i <= n; i++) {
        primes[i] = true;
    }

    for (uint64_t i = 2; i <= n; i++) {
        if (primes[i]) {
            count++;
            for (uint64_t j = 2 * i; j <= n; j += i) {
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
