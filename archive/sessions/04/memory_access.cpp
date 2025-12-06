#include <ctime>
#include <iostream>

using namespace std;

extern "C" {
void testSequential64(unsigned long a[], unsigned long n);
void testSequential64b(unsigned long a[], unsigned long n);
}

/*
 * The optimizer is too clever! It will eliminate the code!
 *
 * The only way is to return the number.
 */

unsigned long testSequentialSum64(const unsigned long a[], const unsigned long n) {
	auto sum = 0UL;
	for (auto i = 0UL; i < n; i++)
		sum += a[i];
	return sum;
}

unsigned long testSequentialSumSq64(const unsigned long a[], const unsigned long n) {
	auto sum = 0UL;
	for (auto i = 0UL; i < n; i++)
		sum += a[i] * a[i];
	return sum;
}

unsigned long testStrideSum64(const unsigned long a[], const unsigned long stride, const unsigned long n) {
	auto sum = 0UL;
	for (auto j = 0UL; j < stride; j++)
		for (unsigned long i = j; i < n; i += stride)
			sum += a[i];
	return sum;
}

unsigned long testCache(const unsigned long a[], const unsigned long cacheSize) {
	auto sum = 0UL;
	for (auto i = 0UL; i < cacheSize; i++)
		sum += a[i];
	return sum;
}

int main() {
	constexpr unsigned long n = 1'000'000'000ULL;
	const auto a = new unsigned long[n];
	auto t0 = clock();
	for (auto i = 0UL; i < n; i++)
		a[i] = i;
	auto t1 = clock();
	cout << "Initializing array: " << static_cast<double>(t1 - t0) / CLOCKS_PER_SEC << endl;

	t0 = clock();
	auto ans = testSequentialSum64(a, n);
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
	constexpr auto cacheSize = 448U * 1024 / sizeof(unsigned long);
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
