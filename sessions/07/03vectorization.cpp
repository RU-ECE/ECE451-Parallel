#include <immintrin.h>
#include <thread>

using namespace std;

//  a[] = 5, 4, 9, 2, 6, 1, 3, 8
//        0  1  2  3  4  5  6  7
//  a[] = 9, 2, 5, 4, 6, 1, 3, 8

/*[(0,2),(1,3),(4,6),(5,7)]
[(0,4),(1,5),(2,6),(3,7)]
[(0,1),(2,3),(4,5),(6,7)]
[(2,4),(3,5)]
[(1,4),(3,6)]
[(1,2),(3,4),(5,6)]
*/
void order(uint32_t& a, uint32_t& b) {
	if (a > b)
		swap(a, b);
}
/*[(0,2),(1,3),(4,6),(5,7)]         [(0,4),(1,5),(2,6),(3,7)]
[(0,1),(2,3),(4,5),(6,7)]           [(2,4),(3,5)]
[(1,4),(3,6)]                       [(1,2),(3,4),(5,6)]
*/
void sortingnetwork(uint32_t a[], const uint32_t n) {
	for (auto i = 0; i < n; i += 8) {
		order(a[i], a[i + 2]);
		order(a[i + 1], a[i + 3]);
		order(a[i + 4], a[i + 6]);
		///... 19
	}
}

//  a[0] = {9, 1, 2,  6, 8, 11, 3, 5}
//  a[1] = {2, 8, 30, -5, 8, 11, 3, 5}
// +

void sortingnetwork(__m256i a[], const uint32_t n) {
	for (uint32_t i = 0; i + 24 <= n; i += 8) {
		__m256i a0 = _mm256_loadu_si256(a + i); // block 0
		__m256i a1 = _mm256_loadu_si256(a + i + 8); // block 1
		const __m256i a2 = _mm256_loadu_si256(a + i + 16); // block 2

		const __m256i temp = _mm256_min_epi32(a0, a1);
		const __m256i temp2 = _mm256_max_epi32(a0, a1);
		a0 = temp;
		a1 = temp2;

		_mm256_storeu_si256(a + i, a0);
		_mm256_storeu_si256(a + i + 8, a1);
		_mm256_storeu_si256(a + i + 16, a2);
	}
}
