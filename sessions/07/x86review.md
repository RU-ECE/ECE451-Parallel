# Intel Intrinsics, AVX Instructions, in general SIMD

vector width makes vectors n x faster
AVX2 = 256 bit
SSE = 128 bit  xmm
Neon (ARM)  128 bit
AVX512 = 512 bit
new ARM = ??? 512, 1024 (Variable length)
Apple IS NOT supporting SVE

```cpp
#include <immintrin.h>


_m256 a;   // ymm00  must know there are 16


void swap(_m256i a, _m256i b) {
// assume that you magically have a,b
_m256i temp = a;
a = __m256_min_eps(a, b);
b = __m256_max_eps(temp, b);
}

                              debug stack
                               x    x
rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8, r9, r10, r11, r12, r13, r14, r15


14 available registers

contrast with ARM: X0, X1, ...X30

AVX2 ymm00 ymm01 ymm15 ..

AVX512 zmm00, zmm31
```


// sum 1/(i*i*i)
double sum(int n) {
  double s = 0;
  for (double i = 1; i <= n; i++)
    s += 1.0 / (i*i*i);
}


// sum 1/(i*i*i)
double sum(int n) {
  _m256d s = __m256_setzero();
  _m256d one = __m256_loadu_pd(1,1,1,1);
  _m256d iv = __m256_loadu_pd(1, 2, 3, 4);
  _m256d four = __m256_loadu_pd(4, 4, 4, 4);
  for (double i = 1; i <= n; i+=4) {
    __m256 term = __m256_mul_pd(__m256_mul_pd(iv, iv),iv) ;
    term = __m256_div_pd(one, term); // 1/(i*i*i)
    s = __m256_add_pd(s, term);    
    iv = __m256_add_pd(iv, four); // 1+4, 2+4, 3+4, 4+4
  }
