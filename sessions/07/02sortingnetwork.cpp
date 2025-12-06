#include <immintrin.h>

// there is no good way that I know to make this work well in C++
// C++ pass by reference will force these values into memory
void sort8(__m256i& a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h) {}

/*
https://bertdobbelaere.github.io/sorting_networks.html#N8L19D6
[(0,2),(1,3),(4,6),(5,7)]
[(0,4),(1,5),(2,6),(3,7)]
[(0,1),(2,3),(4,5),(6,7)]
[(2,4),(3,5)]
[(1,4),(3,6)]
[(1,2),(3,4),(5,6)]
*/

/**
 *   a1 b1 c1 d1 e1 f1 g1 h1
 *   a2 b2 c2 d2 e2 f2 g2 h2
 *   a3 b3 c3 d3 e3 f3 g3 h3
 *   a4 b4 c4 d4 e4 f4 g4 h4
 *   a5 b5 c5 d5 e5 f5 g5 h5
 *   a6 b6 c6 d6 e6 f6 g6 h6
 *   a7 b7 c7 d7 e7 f7 g7 h7
 *   a8 b8 c8 d8 e8 f8 g8 h8
 *
 * transpose to:
 *
 *  a1 a2 a3 a4 a5 a6 a7 a8
 *  b1 b2 b3 b4 b5 b6 b7 b8
 *  c1 c2 c3 c4 c5 c6 c7 c8
 *  d1 d2 d3 d4 d5 d6 d7 d8
 *  e1 e2 e3 e4 e5 e6 e7 e8
 *  f1 f2 f3 f4 f5 f6 f7 f8
 *  g1 g2 g3 g4 g5 g6 g7 g8
 *  h1 h2 h3 h4 h5 h6 h7 h8
 *
 * example, after transpose
 *  xmm0 = [1, 5, 9, 10, 11, 15, 19, 25]
 *             i
 *  xmm1 = [2, 4, 9, 12, 14, 16, 22, 126]
 *             j
 * //1  2 4  5 9 9 10 12


 *  xmm0 = [1, 2, 3, 4, 5, 7, 9, 10]
 *                                  i
 *  xmm1 = [11, 15, 19, 25, 46, 61, 76, 82, 126]
 *             j
 * //1  2 3  4 5 7 9 10

 suppose we have:
 xmm0 = [1   5 ...
 xmm1 = [2   4 ...]
output: 1 2 4 5
*
 */
void transpose(__m256i& a, __m256i& b, __m256i& c, __m256i& d, __m256i& e, __m256i& f, __m256i& g, __m256i& h) {
	const auto temp1 = _mm256_unpacklo_epi32(a, b);
	const auto temp2 = _mm256_unpackhi_epi32(a, b);
	const auto temp3 = _mm256_unpacklo_epi32(c, d);
	const auto temp4 = _mm256_unpackhi_epi32(c, d);
	const auto temp5 = _mm256_unpacklo_epi32(e, f);
	const auto temp6 = _mm256_unpackhi_epi32(e, f);
	const auto temp7 = _mm256_unpacklo_epi32(g, h);
	const auto temp8 = _mm256_unpackhi_epi32(g, h);
	a = _mm256_unpacklo_epi64(temp1, temp3);
	b = _mm256_unpackhi_epi64(temp1, temp3);
	c = _mm256_unpacklo_epi64(temp2, temp4);
	d = _mm256_unpackhi_epi64(temp2, temp4);
	e = _mm256_unpacklo_epi64(temp5, temp7);
	f = _mm256_unpackhi_epi64(temp5, temp7);
	g = _mm256_unpacklo_epi64(temp6, temp8);
	h = _mm256_unpackhi_epi64(temp6, temp8);
}

void sortingnetwork8(const unsigned int arr[], const unsigned int n) {
	for (auto i = 0U; i < n; i += 64) {
		auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + i));
		auto b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + i + 8));
		auto c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + i + 16));
		auto d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + i + 24));
		const auto e = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + i + 32));
		const auto f = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + i + 40));
		const auto g = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + i + 48));
		const auto h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + i + 56));
		// a = [5, 1, 10, 2, 11, 14, 3, 6]
		// c = [6, 2, 9,  3, 8, 6, 4, 7]
		// temp[5, 1, 9, 2,  8, 6, 3, 6]
		// cm = [6, 2, 10, 3, 11, 14, 4, 7]
		{
			const auto temp = _mm256_min_epi32(a, c); // order all pairs in a,c
			c = _mm256_max_epi32(a, c);
			a = temp;
		}
		{
			// to avoid divergence, use vector instructions
			const auto temp = _mm256_min_epi32(b, d); // order all pairs in b,d
			d = _mm256_max_epi32(b, d);
			b = temp;
		}
		//...
		_mm256_storeu_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(arr + i)), a);
		_mm256_storeu_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(arr + i + 8)), b);
		_mm256_storeu_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(arr + i + 16)), c);
		_mm256_storeu_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(arr + i + 24)), d);
		_mm256_storeu_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(arr + i + 32)), e);
		_mm256_storeu_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(arr + i + 40)), f);
		_mm256_storeu_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(arr + i + 48)), g);
		_mm256_storeu_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(arr + i + 56)), h);
	}
}

/**
 *   5, 1, 10, 2, 11, 14, 3, 6,    6, 2, 9,  3, 8, 6, 4, 7
 *   [1 5] [2 10] [11 14] [3 6]
 *    i     j
		  i   j
 *    [1 2 5  10]
 *
 *
 *
 */
