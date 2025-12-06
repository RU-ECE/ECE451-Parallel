#include <iostream>

using namespace std;

class Matrix {
	unsigned int rows;
	unsigned int cols;
	double* data;

public:
	Matrix(const unsigned int r, const unsigned int c, const double v) :
		rows(r), cols(c), data(new double[rows * cols]) { // initializer list
		for (auto i = 0UL; i < rows * cols; i++)
			data[i] = v;
	}

	~Matrix() {
		delete[] data; // it is implementation-defined what happens if you don't delete []
	}

	// copy constructor only happens to uninitialized memory
	Matrix(const Matrix& orig) : rows(orig.rows), cols(orig.cols), data(new double[rows * cols]) {
		for (auto i = 0UL; i < rows * cols; i++)
			data[i] = orig.data[i];
	}

	// operator = must first give back the memory you already have, then copy
	Matrix& operator=(Matrix copy) {
		rows = copy.rows;
		cols = copy.cols;
		swap(data, copy.data);
		return *this;
	}

	friend Matrix operator+(const Matrix& a, const Matrix& b) {
		if (a.rows != b.rows || a.cols != b.cols)
			throw invalid_argument("Matrix sizes don't match");
		Matrix ans(a.rows, a.cols, 0.0); // this is inefficient, creates matrix of all zeros
		for (auto i = 0UL; i < a.rows * a.cols; i++)
			ans.data[i] = a.data[i] + b.data[i];
		return ans;
	}

	friend ostream& operator<<(ostream& os, const Matrix& m) {
		for (auto i = 0UL, c = 0UL; i < m.rows; i++) {
			for (auto j = 0UL; j < m.cols; j++, c++)
				os << m.data[c] << " ";
			os << endl;
		}
		return os;
	}
};
int main() {
	// int a = 2, b = 3, c = 4;
	// a = b = c;
	const Matrix m1(4, 3, 0.0);
	Matrix m2(4, 3, 0.0);
	Matrix m9(2, 2, 0.0);
	m2 = m9 = m1;
	const Matrix m5 = m1 + m2; // add(m1,m2)
	cout << m5 << endl;
	return 0;
}
