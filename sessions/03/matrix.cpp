#include <chrono>
#include <iostream>
#include <memory.h>

using namespace std;
using namespace chrono;

class matrix {
	unsigned int rows, cols;
	float* m;

	// this private constructor is faster, but dangerous (uninitialized)
	// used in functions where we are going to write to all the elements.
	// no reason to write zeros first, only to immediately replace them.
	matrix(const unsigned int rows, const unsigned int cols, const char* internal) :
		rows(rows), cols(cols), m(new float[rows * cols]) {}

public:
	matrix(const unsigned int rows, const unsigned int cols) : rows(rows), cols(cols), m(new float[rows * cols]) {
		memset(m, 0, rows * cols * sizeof(float));
	}
	~matrix() { delete[] m; }
	// copy constructor: needed to avoid crashing on a copy
	// (default implementation does not work for pointers)
	matrix(const matrix& orig) : rows(orig.rows), cols(orig.cols), m(new float[orig.rows * orig.cols]) {
		memcpy(m, orig.m, rows * cols * sizeof(float));
	}
	// support copying with operator = also, which differs from copy
	// because the existing object has to be wiped out
	// this C++11 approach is called copy-and-swap
	matrix& operator=(matrix copy) {
		this->rows = copy.rows;
		this->cols = copy.cols;
		swap(this->m, copy.m);
		return *this;
	}
	// the move constructor, avoiding a copy when the object being
	// copied is a temp and going away anyway. Just steal the memory
	// from the dying object.
	matrix(matrix&& orig) noexcept : rows(orig.rows), cols(orig.cols), m(orig.m) { orig.m = nullptr; }

	// this allows changing an element m(i,j) = 5.2;
	float& operator()(const unsigned int r, const unsigned int c) { return m[r * rows + c]; }

	// this is the readonly version: cout << m(i,j)
	float operator()(const unsigned int r, const unsigned int c) const { return m[r * rows + c]; }

	friend matrix mult(const matrix& a, const matrix& b) {
		matrix ans(a.rows, b.cols, "uninitialized");
		for (auto k = 0U, c = 0U; k < a.rows; k++) {
			for (auto j = 0U; j < b.cols; j++) {
				auto dot = 0.0f;
				const auto ap = a.m + k * a.rows;
				auto bp = b.m + j;
				for (auto i = 0U; i < a.cols; i++, bp += b.cols)
					dot += ap[i] * *bp;
				ans.m[c++] = dot;
			}
		}
		return ans;
	}

	friend matrix operator*(const matrix& a, const matrix& b) {
		matrix ans(a.rows, b.cols, "uninitialized");
		for (auto k = 0U; k < a.rows; k++) {
			for (auto j = 0U; j < b.cols; j++) {
				ans(k, j) = 0;
				for (auto i = 0U; i < a.cols; i++)
					ans(k, j) += a(k, i) * b(i, j);
			}
		}
		return ans;
	}
	static matrix ident(const unsigned int n) {
		matrix ans(n, n, "uninitialized");
		for (auto i = 0U; i < n * n; i += n + 1)
			ans.m[i] = 1;
		return ans;
	}
	friend ostream& operator<<(ostream& s, const matrix& mat) {
		for (auto i = 0U; i < mat.rows; i++) {
			for (auto j = 0U; j < mat.cols; j++)
				s << mat(i, j) << '\t';
			s << endl;
		}
		return s;
	}
};
void test_correctness() {
	auto a = matrix::ident(4);
	auto b = matrix::ident(4);
	auto c = a * b;
	cout << c << endl; // should print 4x4 identity

	a(0, 1) = 2;
	a(2, 2) = 3;
	b(1, 0) = 4;
	b(2, 1) = 5;
	c = a * b;
	cout << c << endl;
	/*
	 *		1 2 0 0			1 0 0 0			9 2 0 0
	 * A =	0 1 0 0		B = 4 1 0 0		C = 4 1 0 0
	 *		0 0 3 0			0 5 1 0			0 0 3 0
	 *		0 0 0 3			0 0 0 1			0 0 0 3
	 */
}

void benchmark() {
	auto t0 = high_resolution_clock::now();
	const auto a = matrix::ident(1024);
	const auto b = matrix::ident(1024);
	auto c = a * b;
	auto t1 = high_resolution_clock::now();
	auto elapsed = duration_cast<microseconds>(t1 - t0);
	cout << "redundant elapsed: " << elapsed.count() << "usec\n";

	t0 = high_resolution_clock::now();
	c = mult(a, b);
	t1 = high_resolution_clock::now();
	elapsed = duration_cast<microseconds>(t1 - t0);
	cout << "optimized elapsed: " << elapsed.count() << "usec\n";
}

int main() {
	test_correctness();
	benchmark();
	return 0;
}
