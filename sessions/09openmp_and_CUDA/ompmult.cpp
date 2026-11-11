#include <omp.h>
#include "matrix.hpp"

void mult_sequential_fastdot_simd(const matrix& a, const matrix& b, matrix& c) {
	if (a.rows != b.cols)
		throw "size mismatch";
  c = 0; // set them all to zero

	for (uint32_t i = 0; i < a.rows; i++) 
		for (uint32_t j = 0; j < b.cols; j++) {
			double dot = 0;
#pragma omp simd reduction(+:dot)
			for (uint32_t k = 0; k < a.cols; k++)
				dot +=  a(i,k) * b(k, j);
			c(i,j) = dot;
		}
}
