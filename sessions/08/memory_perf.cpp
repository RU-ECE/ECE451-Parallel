#include <chrono>
#include <iostream>

using namespace std;
using namespace chrono;

extern "C" {
void read_1byte_sequentially(const char a[], unsigned long n);
void read_1byte_sequentially_b(const char a[], unsigned long n);
void read_8byte_sequentially(const unsigned long a[], unsigned long n);
void read_8byte_skip(const unsigned long a[], unsigned long n, unsigned long skip);
}

/**
 * Read memory sequentially to measure memory performance.
 */
void read_memory(const char a[], const unsigned long n) {
	for (auto i = 0UL; i < n; i++)
		volatile char sink = a[i];
}

/**
 * Read memory sequentially to measure memory performance
 * 64 bits at a time.
 * THis will be faster, but not as much as you might think
 * Each read will take 1/8 the operations
 */
void read_memory64(const unsigned long a[], const unsigned long n) {
	for (auto i = 0UL; i < n; i++)
		volatile auto sink = a[i];
}

/**
 * Read memory with a given skip to measure memory performance.
 * This will be slower as the skip increases.
 */
void read_memory_skip(const unsigned long a[], const unsigned long n, const unsigned long skip) {
	for (auto j = 0UL; j < skip; j++)
		for (auto i = j; i < n; i += skip)
			volatile auto sink = a[i];
}

int main() {
	constexpr auto n = 200'000'000UL;
	constexpr auto n8 = n * 8;
	constexpr auto num_trials = 5UL;
	const auto a = new char[n8]; // what is in here???
	auto t0 = high_resolution_clock::now();
	read_memory(a, n8);
	auto t1 = high_resolution_clock::now();
	cout << "reading bytes cold: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
	for (auto trials = 0U; trials < num_trials; trials++) {
		t0 = high_resolution_clock::now();
		read_memory(a, n8);
		t1 = high_resolution_clock::now();
		cout << "reading bytes warm: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
	}
	for (auto trials = 0U; trials < num_trials; trials++) {
		t0 = high_resolution_clock::now();
		read_1byte_sequentially(a, n8);
		t1 = high_resolution_clock::now();
		cout << "asm reading bytes: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
	}
	for (auto trials = 0U; trials < num_trials; trials++) {
		t0 = high_resolution_clock::now();
		read_1byte_sequentially_b(a, n8);
		t1 = high_resolution_clock::now();
		cout << "asm reading bytes b: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
	}
	delete[] a;
	const auto b = new unsigned long[n];
	t0 = high_resolution_clock::now();
	read_memory64(b, n);
	t1 = high_resolution_clock::now();
	cout << "reading 64-bit words cold: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
	for (auto trials = 0U; trials < num_trials; trials++) {
		t0 = high_resolution_clock::now();
		read_memory64(b, n);
		t1 = high_resolution_clock::now();
		cout << "reading 64-bit words warm: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds"
			 << endl;
	}
	for (auto trials = 0U; trials < num_trials; trials++) {
		t0 = high_resolution_clock::now();
		read_8byte_sequentially(b, n);
		t1 = high_resolution_clock::now();
		cout << "asm 64-bit words: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
	}
	for (auto i = 2UL; i <= 1024; i *= 2) {
		t0 = high_resolution_clock::now();
		read_memory_skip(b, n, i);
		t1 = high_resolution_clock::now();
		cout << "reading memory skip " << i << ": " << duration_cast<microseconds>(t1 - t0).count() << " microseconds"
			 << endl;
	}
	for (auto i = 2UL; i <= 1024; i *= 2) {
		t0 = high_resolution_clock::now();
		read_8byte_skip(b, n / i, i);
		t1 = high_resolution_clock::now();
		cout << "reading memory skip " << i << ": " << duration_cast<microseconds>(t1 - t0).count() << " microseconds"
			 << endl;
	}
	delete[] b;
	return 0;
}
