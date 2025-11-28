#include <cstdint>
#include <iostream>
#include <time.h>

using namespace std;

extern "C" {
void testSequential64(uint64_t a[], uint64_t n);
void testSequential64b(uint64_t a[], uint64_t n);
}


/*
 * The optimizer is too clever! It will eliminate the code!
 *
 * The only way is to return the number
 */
uint64_t testSequentialSum64(uint64_t a[], const uint64_t n) {
	uint64_t sum = 0;
	for (uint64_t i = 0; i < n; i++)
		sum += a[i];
	return sum;
}

uint64_t testSequentialSumSq64(uint64_t a[], const uint64_t n) {
	uint64_t sum = 0;
	for (uint64_t i = 0; i < n; i++)
		sum += a[i] * a[i];
	return sum;
}

uint64_t testStrideSum64(uint64_t a[], const uint64_t stride, const uint64_t n) {
	uint64_t sum = 0;
	for (uint64_t j = 0; j < stride; j++)
		for (uint64_t i = j; i < n; i += stride)
			sum += a[i];
	return sum;
}

uint64_t testCache(volatile uint64_t a[], const uint64_t cacheSize) {
	uint64_t sum = 0;
	for (uint64_t i = 0; i < cacheSize; i++)
		sum += a[i];
	return sum;
}

int main() {
	constexpr uint64_t n = 1'000'000'000ULL;
	const auto a = new uint64_t[n];
	clock_t t0 = clock();
	for (uint64_t i = 0; i < n; i++)
		a[i] = i;
	clock_t t1 = clock();
	cout << "Initializing array: " << static_cast<double>(t1 - t0) / CLOCKS_PER_SEC << endl;

	t0 = clock();
	uint64_t ans = testSequentialSum64(a, n);
	t1 = clock();
	cout << "C++ seq sum add:  " << static_cast<double>(t1 - t0) / CLOCKS_PER_SEC << " result=" << ans << endl;

	t0 = clock();
	ans = testSequentialSumSq64(a, n);
	t1 = clock();
	cout << "C++ seq sum sq add:  " << static_cast<double>(t1 - t0) / CLOCKS_PER_SEC << " result=" << ans << endl;

	t0 = clock();
	testSequential64(a, n);
	t1 = clock();
	cout << "asm seq read:      " << static_cast<double>(t1 - t0) / CLOCKS_PER_SEC << endl;

	t0 = clock();
	testSequential64b(a, n);
	t1 = clock();
	cout << "asm seq read nocmp:" << static_cast<double>(t1 - t0) / CLOCKS_PER_SEC << endl;

	// for actual cache replacement strategies, see:
	// https://en.wikipedia.org/wiki/Cache_replacement_policies
	constexpr uint32_t cacheSize = 448 * 1024 / sizeof(uint64_t);
	t0 = clock();
	testSequential64(a, cacheSize);
	t1 = clock();
	cout << "C++ test cache size " << cacheSize << ": " << static_cast<double>(t1 - t0) / CLOCKS_PER_SEC << endl;

	t0 = clock();
	testCache(a, 2 * cacheSize);
	t1 = clock();
	cout << "C++ test cache size " << cacheSize << ": " << static_cast<double>(t1 - t0) / CLOCKS_PER_SEC << endl;


	t0 = clock();
	ans = testSequentialSumSq64(a, n);
	t1 = clock();
	cout << "elapsed time: " << static_cast<double>(t1 - t0) / CLOCKS_PER_SEC << " result=" << ans << endl;

	for (auto stride = 1; stride <= 1024; stride *= 2) {
		t0 = clock();
		ans = testStrideSum64(a, stride, n);
		t1 = clock();
		cout << "stride: " << stride << ": " << static_cast<double>(t1 - t0) / CLOCKS_PER_SEC << " result=" << ans
			 << endl;
	}
	delete[] a;
}
