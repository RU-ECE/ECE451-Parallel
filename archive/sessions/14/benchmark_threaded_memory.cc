#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;
using namespace chrono;

uint64_t test_r_sequential(uint64_t a[], int n, int stride);
uint64_t test_rw_sequential(uint64_t a[], int n, int stride);
uint64_t test_r_stride(uint64_t a[], int n, int stride);
uint64_t test_r_cache(uint64_t a[], int n, int stride);
uint64_t test_r_cache_pipelineproblems(uint64_t a[], int n, int stride);
uint64_t test_r_cache3(uint64_t a[], int n, int stride);

uint64_t min(const vector<uint64_t>& v) {
	uint64_t m = v[0];
	for (const auto x : v)
		if (x < m)
			m = x;
	return m;
}


template <typename Func>
void testthreads(const char msg[], Func f, int n, const int numthreads, int stride) {
	auto a = new uint64_t[n]; // lots o zeros...
	// compute the start and end time of the benchmark
	vector<uint64_t> benchmarks;
	cout << msg << " threads=" << numthreads;

	for (uint32_t trial = 0; trial < 5; trial++) {
		vector<thread*> threads;
		auto t0 = high_resolution_clock().now();

		for (auto i = 0; i < numthreads; i++) {
			thread* t = new thread(f, a, n, stride);
			t->join();
			threads.push_back(t);
		}
		auto t1 = high_resolution_clock().now();
		const auto elapsed = duration_cast<milliseconds>(t1 - t0).count();
		benchmarks.push_back(elapsed);
		for (const auto t : threads)
			delete t;
	}
	cout << " elapsed=" << min(benchmarks) << "ms\n";
}

int main() {
	for (auto threads = 1; threads <= 8; threads *= 2) {
		constexpr auto n = 400'000'000;
		testthreads("seq read", test_r_sequential, n, threads, 1);
		testthreads("seq write", test_rw_sequential, n, threads, 1);
		testthreads("stride32", test_r_stride, n, threads, 32);
		testthreads("stride1024", test_r_stride, n, threads, 1024);
		testthreads("cache read", test_r_cache, n, threads, 1);
		testthreads("cache badpipe", test_r_cache_pipelineproblems, n, threads, 1);
		testthreads("cache read3", test_r_cache3, n, threads, 1);
		cout << endl;
	}
}


// sum every element of an array
uint64_t test_r_sequential(uint64_t a[], const int n, int stride) {
	uint64_t sum = 0;
	for (uint64_t i = 0; i < n; i++)
		sum += a[i];
	return sum;
}

// increment every element of an array (read and write)
uint64_t test_rw_sequential(uint64_t a[], const int n, int stride) {
	for (uint64_t i = 0; i < n; i++)
		a[i]++;
	return 0;
}


// sum elements out of order
uint64_t test_r_stride(uint64_t a[], const int n, const int stride) {
	uint64_t sum = 0;
	for (uint64_t j = 0; j < stride; j++)
		for (uint64_t i = j; i < n; i += stride)
			sum += a[i];
	return sum;
}

// read out of cache
uint64_t test_r_cache(uint64_t a[], const int n, int stride) {
	uint64_t sum = 0;
	for (uint64_t i = 0, j = 0; i < n; i += 4) {
		sum += a[j]; // cached because j never changes
		sum += a[j + 1];
		sum += a[j + 2];
		sum += a[j + 3];
	}
	return sum;
}

// read out of cache
uint64_t test_r_cache_pipelineproblems(uint64_t a[], const int n, int stride) {
	uint64_t sum = 0;
	for (uint64_t i = 0, j = 0; i < n; i++) {
		sum += a[j++]; // cached because 0 <= j < 8
		if (j > 8)
			j = 0;
	}
	return sum;
}

// read out of cache without the stinking if
uint64_t test_r_cache3(uint64_t a[], const int n, int stride) {
	uint64_t sum = 0;
	for (uint64_t i = 0, j = 0; i < n; i++) {
		sum += a[j];
		j = j + 1 & 7; // same as if, but faster
	}
	return sum;
}
