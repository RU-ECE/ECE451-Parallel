#include <immintrin.h>
#include <thread>

using namespace std;

// sorting

void sort8cols(__m256i& a, __m256i& b, __m256i& c, __m256i& d, __m256i& e, __m256i& f, __m256i& g, __m256i& h) {
	auto cond_swap = [](__m256i& x, __m256i& y) {
		const auto mask = _mm256_cmpgt_epi32(x, y), // mask: all ones where x > y
			tmp = x, x_new = _mm256_blendv_epi8(x, y, mask), y_new = _mm256_blendv_epi8(y, tmp, mask);
		x = x_new;
		y = y_new;
	};
	cond_swap(a, b);
	cond_swap(b, c);
	cond_swap(c, d);
	cond_swap(d, e);
	cond_swap(e, f);
	cond_swap(f, g);
	cond_swap(g, h);
}

void sort16cols(__m256i a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h) {}

void transpose8() {}

void merge(__m256i a, __m256i b, __m256i c, __m256i d) {}

void sort(int a[], const int n) {
	for (auto i = 0; i < n; i++) {
		for (auto j = i + 1; j < n; j++) {
			if (a[i] > a[j]) {
				const auto temp = a[i];
				a[i] = a[j];
				a[j] = temp;
			}
		}
	}
}
