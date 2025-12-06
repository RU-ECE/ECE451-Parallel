#include <chrono>
#include <iostream>

using namespace std;
using namespace chrono;

extern "C" {
void read_one(unsigned long* data, int n);
void read_memory_scalar(unsigned long* data, int n);
void read_memory_sse(unsigned long* data, int n);
void read_one_avx(unsigned long* data, int n);
void read_memory_avx(unsigned long* data, int n);
void read_memory_sse_unaligned(unsigned long* data, int n);
void read_memory_avx_unaligned(unsigned long* data, int n);
void read_memory_every2(unsigned long* data, int n);
void read_memory_everyk(unsigned long* data, int n, int k);
void write_one(unsigned long* data, int n);
void write_one_avx(unsigned long* data, int n);
void write_memory_scalar(unsigned long* data, int n);
void write_memory_sse(unsigned long* data, int n);
void write_memory_avx(unsigned long* data, int n);
}

// for information on how C++ passes parameters to functions, see:
// https://en.wikipedia.org/wiki/X86_calling_conventions (search for linux)

template <typename Func, typename... Args>
void benchmark(const char name[], Func read_memory, unsigned long* p, int n, Args... args) {
	const auto start = high_resolution_clock::now();
	read_memory(p, n, args...);
	const auto end = high_resolution_clock::now();
	const duration<double> elapsed = end - start;
	cout << name << " took " << elapsed.count() << " seconds " << elapsed.count() / n * 1e9 << " ns per element"
		 << endl;
}

int main() {
	constexpr auto n = 1024UL * 1024 * 512; // 4 GB, 512 million 64-bit words
	const auto p = static_cast<unsigned long*>(aligned_alloc(32, n * sizeof(unsigned long))); // allocate

	benchmark("warmup (disregard)", read_memory_scalar, p, n);
	benchmark("read_one", read_one, p, n);
	benchmark("write_one", write_one, p, n);
	benchmark("read_memory_scalar", read_memory_sse_unaligned, p, n);
	benchmark("read_memory_sse", read_memory_sse, p, n);
	benchmark("read_memory_avx", read_memory_avx, p, n);
	benchmark("read_memory_sse_unaligned", read_memory_sse_unaligned, p, n);
	benchmark("read_memory_avx_unaligned", read_memory_avx_unaligned, p, n);
	benchmark("read_memory_every2", read_memory_every2, p, n);
	benchmark("read_memory_every 4", read_memory_everyk, p, n, 4);
	benchmark("read_memory_every 8", read_memory_everyk, p, n, 8);
	benchmark("read_memory_everyk 16", read_memory_everyk, p, n, 16);
	benchmark("read_memory_everyk 32", read_memory_everyk, p, n, 32);
	benchmark("read_memory_everyk 64", read_memory_everyk, p, n, 64);
	benchmark("read_memory_everyk 128", read_memory_everyk, p, n, 128);
	benchmark("read_memory_everyk 256", read_memory_everyk, p, n, 256);
	benchmark("read_memory_everyk 512", read_memory_everyk, p, n, 512);
	benchmark("read_memory_everyk 1024", read_memory_everyk, p, n, 1024);
	benchmark("read_memory_everyk 1M", read_memory_everyk, p, n, 1024 * 1024);
	benchmark("read_one_avx", read_one_avx, p, n);
	benchmark("write_one_avx", write_one_avx, p, n);
	benchmark("write_memory_scalar", write_memory_scalar, p, n);
	benchmark("write_memory_avx", write_memory_avx, p, n);
	free(p);
	return 0;
}
