#include <iostream>

using namespace std;

class Matrix {
	unsigned int rows;
	unsigned int cols;
	double* data;
	Matrix(const unsigned int r, const unsigned int c) : rows(r), cols(c), data(new double[rows * cols]) {}

public:
	Matrix(const unsigned int r, const unsigned int c, const double v) :
		rows(r), cols(c), data(new double[rows * cols]) { // initializer list
		for (auto i = 0U; i < rows * cols; i++)
			data[i] = v;
	}

	~Matrix() {
		delete[] data; // it is implementation-defined what happens if you don't delete []
	}

	// copy constructor only happens to uninitialized memory
	Matrix(const Matrix& orig) : rows(orig.rows), cols(orig.cols), data(new double[rows * cols]) {
		for (auto i = 0U; i < rows * cols; i++)
			data[i] = orig.data[i];
	}

	// operator = must first give back the memory you already have, then copy
	Matrix& operator=(Matrix copy) {
		rows = copy.rows;
		cols = copy.cols;
		swap(data, copy.data);
		return *this;
	}

	Matrix(Matrix&& orig) noexcept : rows(orig.rows), cols(orig.cols), data(orig.data) {
		orig.data = nullptr; // rob the dying object of its memory
	}

	friend Matrix operator+(const Matrix& a, const Matrix& b) {
		if (a.rows != b.rows || a.cols != b.cols)
			throw invalid_argument("Matrix sizes don't match");
		Matrix ans(a.rows, a.cols, 0.0); // this is inefficient, creates matrix of all zeros
		for (auto i = 0U; i < a.rows * a.cols; i++)
			ans.data[i] = a.data[i] + b.data[i];
		return ans;
	}

	double operator()(const int r, const int c) const { // this is READONLY
		return data[r * cols + c];
	}

	double& operator()(const int r, const int c) { // this operator can change m(r,c)
		return data[r * cols + c];
	}

	friend Matrix operator*(const Matrix& a, const Matrix& b) {
		if (a.cols != b.rows)
			throw invalid_argument("Matrix sizes don't match");
		Matrix ans(a.rows, b.cols);
		for (auto i = 0U; i < a.rows; i++) {
			auto dot = 0.0;
			for (auto j = 0U; j < b.cols; j++) {
				for (auto k = 0U; k < a.cols; k++)
					dot += a(i, k) * b(k, j);
				ans(i, j) = dot;
			}
		}
		return ans;
	}

	friend ostream& operator<<(ostream& os, const Matrix& m) {
		for (auto i = 0U, c = 0U; i < m.rows; i++) {
			for (auto j = 0U; j < m.cols; j++, c++)
				os << m.data[c] << " ";
			os << endl;
		}
		return os;
	}

	static Matrix ident(const unsigned int n) {
		Matrix ans(n, n, 0.0);
		for (auto i = 0U; i < n; i++)
			ans(i, i) = 1.0;
		return ans;
	}
};
int main() {
	Matrix m1(4, 3, 0.0);
	Matrix m2(4, 3, 0.0);
	cout << m1(2, 3) << endl;
	m1(2, 3) = 2.0;
	m2(3, 3) = -2.0;
	const auto m3 = Matrix::ident(4);
	auto m4 = Matrix::ident(4);
	m4(3, 3) = -2.0;
	m4(1, 2) = 9;
	const auto m5 = m1 + m2;
	const auto m6 = m3 * m4;
	cout << m5 << endl;
	cout << m6 << endl;
	return 0;
}
