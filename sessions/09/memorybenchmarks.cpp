#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <thread>

using namespace std;
using namespace chrono;

void memcopy1(char* dest, const char* src, uint64_t num_bytes);
void memcopy2(uint64_t* dest, const uint64_t* src, uint64_t num_bytes);
void memcopy3(__m256i* dest, const __m256i* src, uint64_t num_bytes);
void memcopy4(__m256i* dest, const __m256i* src, uint64_t num_bytes);


int main() {
	constexpr int n = 1024 * 1024 * 128;
	auto src = static_cast<uint64_t*>(aligned_alloc(32, n * sizeof(uint64_t)));
	auto dest = static_cast<uint64_t*>(aligned_alloc(32, n * sizeof(uint64_t)));
	// first warm up
	auto t0 = high_resolution_clock::now();
	memcopy1(reinterpret_cast<char*>(dest), reinterpret_cast<const char*>(src), n);
	auto t1 = high_resolution_clock::now();
	cout << "cold: memcopy1 took " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;

	t0 = high_resolution_clock::now();
	memcopy1(reinterpret_cast<char*>(dest), reinterpret_cast<const char*>(src), n);
	t1 = high_resolution_clock::now();
	cout << "warm: memcopy1 took " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;

	t0 = high_resolution_clock::now();
	memcopy2(dest, src, n);
	t1 = high_resolution_clock::now();
	cout << "memcopy2 took " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
	t0 = high_resolution_clock::now();
	memcopy3(reinterpret_cast<__m256i*>(dest), reinterpret_cast<__m256i*>(src), n);
	t1 = high_resolution_clock::now();
	cout << "memcopy3 took " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
	t0 = high_resolution_clock::now();
	memcopy4(reinterpret_cast<__m256i*>(dest), reinterpret_cast<__m256i*>(src), n);
	t1 = high_resolution_clock::now();
	cout << "memcopy4 took " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;

	t0 = high_resolution_clock::now();
	memcpy(dest, src, n);
	t1 = high_resolution_clock::now();
	cout << "memcpy took " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;

	{
		t0 = high_resolution_clock::now();
		thread thr1(memcopy1, reinterpret_cast<char*>(dest), reinterpret_cast<const char*>(src), n);
		thread thr2(memcopy1, reinterpret_cast<char*>(dest), reinterpret_cast<const char*>(src), n);
		thread thr3(memcopy1, reinterpret_cast<char*>(dest), reinterpret_cast<const char*>(src), n);
		thread thr4(memcopy1, reinterpret_cast<char*>(dest), reinterpret_cast<const char*>(src), n);
		thr1.join();
		thr2.join();
		thr3.join();
		thr4.join();
		t1 = high_resolution_clock::now();
		cout << "threaded memcopy1 took " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
	}
	{
		t0 = high_resolution_clock::now();
		thread thr1(memcopy2, dest, src, n);
		thread thr2(memcopy2, dest, src, n);
		thread thr3(memcopy2, dest, src, n);
		thread thr4(memcopy2, dest, src, n);
		thr1.join();
		thr2.join();
		thr3.join();
		thr4.join();
		t1 = high_resolution_clock::now();
		cout << "threaded memcopy2 took " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
	}

	{
		t0 = high_resolution_clock::now();
		thread thr1(memcpy, static_cast<void*>(dest), static_cast<const void*>(src), n);
		thread thr2(memcpy, static_cast<void*>(dest), static_cast<const void*>(src), n);
		thread thr3(memcpy, static_cast<void*>(dest), static_cast<const void*>(src), n);
		thread thr4(memcpy, static_cast<void*>(dest), static_cast<const void*>(src), n);
		thr1.join();
		thr2.join();
		thr3.join();
		thr4.join();
		t1 = high_resolution_clock::now();
		cout << "threaded memcopy2 took " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
	}


	free(src);
	free(dest);
	return 0;
}
