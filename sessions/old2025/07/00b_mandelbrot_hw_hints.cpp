#include <immintrin.h>

int main() {
    __m256s a = _mm256_set_zero_ps();
    __m256d dx = _mm256_set_ps(0.1); // (0.1, 0.1, 0.1, ...)
    __m256s s = _mm256_set_ps(0, 1, 2, 3, 4, 5, 6, 7);
    dx = _mm256_mul_ps(dx, s); // (0dx, 1dx, 2dx, ....)
    // (r, i),   (r+d, i), (r+2d, i)
for (int i = 0; i < w;  i+= 8)


// there are instructions for multiply, fused multiply-add


/* divergence
  in SIMD, all data is treated THE SAME
   __m256 count = _mm256_setzero_ps();
   __m256 inc = _mm256_set1_ps(1);
   for (i = 0; i < max_iter && abs(z) < 2; i++) {
            z = z*z + c;
        }
        // write the count to the pixels



    x,y      x+d, y      x+2d, y ....
    c = 0    c= 0        c = 0 ...
    c = 1    c= 1        c = 1 ...
    ...    

            STOP!
*/