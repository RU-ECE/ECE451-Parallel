#include <immintrin.h>

void sortnettest(__m256i& ar, __m256i& br, __m256i& cr, __m256i& dr) {
	__m256i a = ar;
	const __m256i b = br;
	__m256i c = cr;
	const __m256i d = dr;
	const __m256i temp = _mm256_min_epi32(a, c); // order all pairs in a,c
	c = _mm256_max_epi32(a, c);
	a = temp;
	ar = a;
	br = b;
	cr = c;
	dr = d;
}

/*
vmovdqa (%rdi),%ymm4
vmovdqa (%rdi),%ymm3
vmovdqa (%rdi),%ymm2
vmovdqa (%rdi),%ymm1
vpminsd %ymm1,%ymm2,%ymm0     # ymm0 = min(ymm1,ymm2)
vpmaxsd %ymm1,%ymm2,%ymm2     # ymm2 = max(ymm1, ymm2)


   8:	c5 fd 6f 16          	vmovdqa (%rsi),%ymm2
   c:	c5 fd 6f 09          	vmovdqa (%rcx),%ymm1
  10:	c4 e2 5d 39 1a       	vpminsd (%rdx),%ymm4,%ymm3
  15:	c4 e2 5d 3d 02       	vpmaxsd (%rdx),%ymm4,%ymm0


*/
