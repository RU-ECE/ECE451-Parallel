#include <iostream>

using namespace std;

class Matrix {
	unsigned int rows = 0;
	unsigned int cols = 0;
	double* data = nullptr;
	Matrix(const unsigned int rows, const unsigned int cols) : rows(rows), cols(cols), data(new double[rows * cols]) {}

public:
	Matrix(const unsigned int r, const unsigned int c, const double v) :
		rows(r), cols(c), data(new double[rows * cols]) {
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
		Matrix ans(a.rows, a.cols); // only the implementor can have the uninitialized one
		for (auto i = 0U; i < a.rows * a.cols; i++)
			ans.data[i] = a.data[i] + b.data[i];
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
};

int main() {
	const Matrix m1(4, 3, 0.0);
	const Matrix m2(4, 3, 0.0);
	const Matrix m5 = m1 + m2;
	cout << m5 << endl;
	return 0;
}
