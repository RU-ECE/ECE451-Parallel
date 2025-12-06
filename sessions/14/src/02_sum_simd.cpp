#include <iostream>

using namespace std;

/*
 * Note: when we compiled with g++ -O3 -fopenmp -o sum_simd sum_simd.cpp this only did SSE to do AVX:
 *		g++ -O3 -mavx2 -fopenmp -o sum_simd sum_simd.cpp
 */

int main() {
	constexpr auto n = 16;
	const auto a = new float[n];
	const auto b = new float[n];
	const auto c = new float[n];
	for (auto i = 0; i < n; i++) {
		a[i] = static_cast<float>(i);
		b[i] = static_cast<float>(i);
	}
#pragma omp parallel for simd
	for (auto i = 0; i < n; i++)
		c[i] = a[i] + b[i];
	for (auto i = 0; i < n; i++)
		cout << c[i] << ' ';
	cout << endl;
	delete[] a;
	delete[] b;
	delete[] c;
}
