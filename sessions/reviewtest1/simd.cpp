#include <immintrin.h>
#include <iostream>

using namespace std;

__m256d sum(const double x[], int n) {
  __m256d s = _m256d_set_zero(); // how many values are in s?  0,0,0,0
  for (int i = 0; i < n; i += 4) {
		__m256d x = _m256d_load(&x[i]); // 1, 2, 3, 4
		s = _m256d_addpd(s, x); // 0+1, 0+2, 0+3, 0+4 (1,2,3,4)
		//                                             5,6,7,8
		//                                             6,8,10,12
	}
	return s;
}

int main() {
	double x[] = {1, 2, 3, 4, 5, 6, 7, 8};
