#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;
using namespace chrono;

unsigned long test_r_sequential(const unsigned long a[], int n, int stride);
unsigned long test_rw_sequential(unsigned long a[], int n, int stride);
unsigned long test_r_stride(const unsigned long a[], int n, int stride);
unsigned long test_r_cache(const unsigned long a[], int n, int stride);
unsigned long test_r_cache_pipelineproblems(const unsigned long a[], int n, int stride);
unsigned long test_r_cache3(const unsigned long a[], int n, int stride);

unsigned long min(const vector<unsigned long>& v) {
	auto m = v[0];
	for (const auto x : v)
		if (x < m)
			m = x;
	return m;
}

template <typename Func>
void testthreads(const char msg[], Func f, int n, const int numthreads, int stride) {
	auto a = new unsigned long[n]; // lots o zeros...
	// compute the start and end time of the benchmark
	vector<unsigned long> benchmarks;
	cout << msg << " threads=" << numthreads;
	for (auto trial = 0; trial < 5; trial++) {
		vector<thread*> threads;
		auto t0 = high_resolution_clock::now();
		for (auto i = 0; i < numthreads; i++) {
			auto* t = new thread(f, a, n, stride);
			t->join();
			threads.push_back(t);
		}
		auto t1 = high_resolution_clock::now();
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
unsigned long test_r_sequential(const unsigned long a[], const int n, int stride) {
	auto sum = 0UL;
	for (auto i = 0; i < n; i++)
		sum += a[i];
	return sum;
}

// increment every element of an array (read and write)
unsigned long test_rw_sequential(unsigned long a[], const int n, int stride) {
	for (auto i = 0; i < n; i++)
		a[i]++;
	return 0;
}

// sum elements out of order
unsigned long test_r_stride(const unsigned long a[], const int n, const int stride) {
	auto sum = 0UL;
	for (auto j = 0; j < stride; j++)
		for (auto i = j; i < n; i += stride)
			sum += a[i];
	return sum;
}

// read out of cache
unsigned long test_r_cache(const unsigned long a[], const int n, int stride) {
	auto sum = 0UL;
	for (auto i = 0, j = 0; i < n; i += 4) {
		sum += a[j]; // cached because j never changes
		sum += a[j + 1];
		sum += a[j + 2];
		sum += a[j + 3];
	}
	return sum;
}

// read out of cache
unsigned long test_r_cache_pipelineproblems(const unsigned long a[], const int n, int stride) {
	auto sum = 0UL;
	for (auto i = 0, j = 0; i < n; i++) {
		sum += a[j++]; // cached because 0 <= j < 8
		if (j > 8)
			j = 0;
	}
	return sum;
}

// read out of cache without the stinking if
unsigned long test_r_cache3(const unsigned long a[], const int n, int stride) {
	auto sum = 0UL;
	for (auto i = 0, j = 0; i < n; i++) {
		sum += a[j];
		j = j + 1 & 7; // same as if, but faster
	}
	return sum;
}
