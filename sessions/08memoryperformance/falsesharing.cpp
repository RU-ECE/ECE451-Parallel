#include <thread>
#include <cstdint>

using namespace std;

volatile uint64_t count[4*8];

/*
	Falsesharing is when one thread writing to a cache line invalidates
	other threads cached values in the same line

	Moral: use registers. If you can't registers, use cache
	Avoid writing if you possibly can.


 */
#if 0
// this was the way to count primes without using memory
void countPrimes(uint64_t a, uint64_t b, uint64_t*res) {
	uint64_t mycount = 0;
	for (uint64_t i = a; i <= b; i++) {
		if (isPrime(i))
			mycount++;
	}
	*res = mycount;
}
#endif

void playwithcounter(int which) {
	for (uint64_t i = 0; i < 200'000'000; i++)
  	count[which]++;
}

int main() {
	thread t0(playwithcounter, 0);
	thread t1(playwithcounter, 8);
	thread t2(playwithcounter, 16);
	thread t3(playwithcounter, 24);
	t0.join();
	t1.join();
	t2.join();
	t3.join();
}
