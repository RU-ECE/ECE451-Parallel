#include "matrix.hpp"

using namespace std;

void mult_sequential(const matrix& a, const matrix& b, matrix& c) {
	if (a.rows != b.cols)
		throw "size mismatch";
  c = 0; // set them all to zero

	for (uint32_t i = 0; i < a.rows; i++) 
		for (uint32_t j = 0; j < b.cols; j++)
			for (uint32_t k = 0; k < a.cols; k++)
				c(i,j) +=  a(i,k) * b(k, j);
}

void mult_sequential_fastdot(const matrix& a, const matrix& b, matrix& c) {
	if (a.rows != b.cols)
		throw "size mismatch";
  c = 0; // set them all to zero

	for (uint32_t i = 0; i < a.rows; i++) 
		for (uint32_t j = 0; j < b.cols; j++) {
			double dot = 0;
			for (uint32_t k = 0; k < a.cols; k++)
				dot +=  a(i,k) * b(k, j);
			c(i,j) = dot;
		}
}

void bench_sequential(const matrix& m1, const matrix& m2, matrix& m3) {
	//	mult_sequential(m1, m2, m3); // m3 = m1*m2
	mult_sequential_fastdot(m1, m2, m3); // m3 = m1*m2
}


#if 0
void test() {
	const int n = 10;
	matrix m1(n,n), m2(n,n), m3(n,n);

	// fill with values

	mul_sequential(m1, m2, m3); // m3 = m1*m2
}
#endif

int main() {
	const int n = 1024;
  matrix m1(n,n), m2(n,n), m3(n,n);
	m1 = 1.5;
	m2 = 1;
	bench_sequential(m1, m2, m3);
}
