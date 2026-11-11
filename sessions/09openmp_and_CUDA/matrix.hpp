#pragma once

#include <iostream>
#include <cmath>

class matrix {
private:
	double* m; // row-major order
	
	/*
		random access: RAS + CAS + 15 gives you 16  burst 64-bit numbers
		next burst in row: CAS + 15 gives you 16 more in the same row
		a00 a01 a02 ...   a0n
		a10 a11 a12 ...   a1n

	 */
	uint32_t rows, cols;

public:
	matrix(uint32_t rows, uint32_t cols) :
		rows(rows), cols(cols), m(new double[rows*cols]) {
		for (uint32_t i = 0; i < rows*cols; i++)
			m[i] = 0;
	}
	~matrix() {
		delete []m;
	}
  matrix(const matrix& orig) = delete;	
  matrix& operator =(const matrix& orig) = delete;	

	double operator ()(uint32_t r, uint32_t c) const { return m[r * cols + c]; }
	double& operator ()(uint32_t r, uint32_t c) { return m[r * cols + c]; }
	void operator=(double v) {
		uint32_t c = 0;
		for (uint32_t i = 0; i < rows; i++)
			for (uint32_t j = 0; j < cols; j++)
				m[c++] = v;
	}
	
	void print() const {
		for (uint32_t i = 0; i < rows; i++) {
			for (uint32_t j = 0; j < cols; j++) {
				std::cout << (*this)(i,j) << ' ';
			}
			std::cout << '\n';
		}
	}
	friend void mult_sequential(const matrix& a, const matrix& b, matrix& c);
  friend void mult_sequential_fastdot(const matrix& a, const matrix& b, matrix& c);
	friend void mult_sequential_fastdot_simd(const matrix& a, const matrix& b, matrix& c);

};
