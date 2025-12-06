auto prime_count = 0;

__global__ void eratosthenes(unsigned int* primes, const unsigned int n) {
	const auto tid = threadIdx.x; // each thread handles one element
	auto count = 0U; // where is this?? Who owns this?
	// 65, 67, 69, 71, 73, 75, 77, 79, 81, 83
	// t0 = 1..63, t1 = 65..127, ...
	// 3, 3*3, 3*5, 3*7, ...
	// 5, 5*5, 5*7, 5*9, ...
	// here's a potentially better idea:
	// let's all cooperate to write to memory[2, 1024]
	// next thread can write [1025...2048]
	// each thread will check numbers tid, tid+1, tid+2, ...
	const auto start = tid * 64 + 1, end = start + 64;
	for (auto i = start; i < end; i += 2) {
		if (primes[i]) { // divergent
			count++;
			for (auto j = i * i; j <= n; j += i)
				primes[j] = 0;
		}
		// all threads NOT PRIME will sleep
	}
}

__global__ void better_eratosthenes(unsigned int* primes, unsigned int start, unsigned int end, unsigned int n) {
	auto tid = threadIdx.x; // each thread handles one element
	auto count = 0U; // where is this?? Who owns this?
	__shared__ localstorage[256];
	unsigned int local[64]; // allocates this in registers?
}

int main() {
	constexpr auto n = 1'000'000;
	const auto primes = new unsigned int[(n + 31) / 32];
	for (auto i = 0; i < n; i++)
		primes[i] = 1;
	eratosthenes<<<(n + 31) / 32, 32>>>(primes, n);
	delete[] primes;
}
