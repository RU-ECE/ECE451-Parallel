#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <memory.h>

/*
 * https://xhad1234.github.io/Parallel-Sort-Merge-Join-in-Peloton/
 *
 * repo: https://github.com/sid1607/avx2-merge-sort
 *
 * remember: as far as I can tell, their code DOESN'T WORK!!!
 *
 * https://www.felixcloutier.com/x86/
 */

__m256i f(const __m256i a, const __m256i b, const __m256i c) { return a * b + c; }

// the if statement is terrible for pipelining
inline void minmax(const int a, const int b, int& minab, int& maxab) {
	if (a > b) {
		minab = b;
		maxab = a;
	} else {
		minab = a;
		maxab = b;
	}
}

void minmax(__m256i& a, __m256i& b) {
	const __m256i temp = _mm256_min_epi32(a, b);
	b = _mm256_max_epi32(a, b);
	a = temp;
}

/*
 * C++ calling convention ON x86 running linux
 *
 * f(int a, int b) -> a = esi, b = edi
 * f(uint64_t a, uint64_t b, uint64_t c, uint64_t d) -> a = rsi, b = rdi, c = r8, d = r9
 * g(double a, double b, double c, double d, double e) -> ymm0, ymm1, ymm2, ymm3, ymm5
 *
 * r(__m256i a, __m256i b, __m256i c, __m256i d) -> a = ymm0, b = ymm1, c = ymm2, d = ymm3
 */

void sort8cols(__m256i a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h) {
	minmax(a, b);
	minmax(c, d);
	minmax(e, f);
	minmax(g, h);
	minmax(a, c);
	//...
}

/*
 * given pointer to 64 consecutive 32-bit integers
 * sort each column of 8
 */
void sort8cols(uint32_t* p) {
	const __m256i a = _mm256_load_si256(reinterpret_cast<__m256i const*>(p));
	// load 8 _m256i registers a..h
	__m256i aout, bout, cout, dout, eout, fout, gout, hout;
	// sort8cols(a,b,c,d,e,f,g,h, aout, bout, cout, dout, eout, fout, gout, hout);
	// store 8 _m256i registers a..h
	_mm256_store_si256(reinterpret_cast<__m256i*>(p), a);
}


__m256i load(const uint32_t* ap) {
	const auto temp = static_cast<uint32_t*>(aligned_alloc(32, 32));
	memcpy(temp, ap, 32);
	__m256i v = _mm256_load_si256(reinterpret_cast<__m256i const*>(temp));
	free(temp);
	return v;
}

extern "C" {
void sort8colsasm(__m256i a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h);
}

int main() {
	const uint32_t a[8] = {57, 58, 59, 60, 61, 62, 63, 64};
	const uint32_t b[8] = {49, 50, 51, 52, 53, 54, 55, 56};
	const uint32_t c[8] = {41, 42, 43, 44, 45, 46, 47, 48};
	const uint32_t d[8] = {33, 34, 35, 36, 37, 38, 39, 40};
	const uint32_t e[8] = {25, 26, 27, 28, 29, 30, 31, 32};
	const uint32_t f[8] = {17, 18, 19, 20, 21, 22, 23, 24};
	const uint32_t g[8] = {9, 10, 11, 12, 13, 14, 15, 16};
	const uint32_t h[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	const __m256i av = load(a);
	const __m256i bv = load(b);
	const __m256i cv = load(c);
	const __m256i dv = load(d);
	const __m256i ev = load(e);
	const __m256i fv = load(f);
	const __m256i gv = load(g);
	const __m256i hv = load(h);
	sort8cols(av, bv, cv, dv, ev, fv, gv, hv);
	// printvec(av); printvec(bv)...
	sort8colsasm(av, bv, cv, dv, ev, fv, gv, hv);
	return 0;
}
