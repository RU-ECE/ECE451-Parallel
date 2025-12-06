#include <thread>

using namespace std;

class Matrix {
	unsigned int rows;
	unsigned int cols;
	double* data;

public:
	Matrix(const unsigned int r, const unsigned int c, const double v) :
		rows(r), cols(c), data(new double[rows * cols]) { // initializer list
		for (auto i = 0U; i < rows * cols; i++)
			data[i] = v;
	}

	~Matrix() {
		delete[] data; // it is implementation-defined what happens if you don't delete []
	}
	// either write the copy constructor, or explicitly delete the copy constructor making a copy illegal
	// Matrix(const Matrix& orig) = delete;
	// Matrix& operator =(const Matrix& orig) = delete;

	Matrix(const Matrix& orig) : rows(orig.rows), cols(orig.cols), data(new double[rows * cols]) {
		for (auto i = 0U; i < rows * cols; i++)
			data[i] = orig.data[i];
	}

#if 0
	// this is the legacy copy constructor still works (but annoying to write copy twice)
	Matrix& operator=(const Matrix& orig) {
		if (this == &orig)
			return *this;
		rows = orig.rows;
		cols = orig.cols;
		delete[] data;
		data = new double[rows * cols];
		for (int i = 0; i < rows * cols; i++)
			data[i] = orig.data[i];
		return *this;
	}
#endif
	// new paradigm: copy and swap
	Matrix& operator=(Matrix copy) {
		rows = copy.rows;
		cols = copy.cols;
		swap(data, copy.data);
		return *this;
	}
};

void f() {
	auto x = 2;
	int a[1024] = {};
	int b[1024 * 1024] = {};
	// THIS WILL CRASH!! too much memory allocated on the stack for modern operating systems
	// this is because of paranoia due to stack smashing attack
}

void g(const Matrix& m2) { // m2 is a COPY of the original matrix m
}

Matrix h() { return Matrix{1, 1, 0.0}; }
int main() {
#if 0
	int* p = new int[100];
	delete [] p;
	delete [] p; // cannot delete a pointer twice
#endif
	const Matrix m(3, 4, 0.0);
	g(m);
	Matrix m2 = m; // copy constructor
	Matrix m3(m); // copy constructor
	Matrix m4(2, 2, 0.0); // create another matrix
	m4 = m3; // assignment operator
	m3 = m3;
#if 0
	f();
	// at compile time, multidimensional arrays are legal
	int a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
	Matrix m1(4, 3, 0.0);
	Matrix m2(4, 4, 0.0);
	m1(2, 3) = 2.0;
	m2(3, 3) = -2.0;
	Matrix m3 = Matrix::ident(4);
	Matrix m4 = Matrix::ident(4);
	m4(3, 3) = -2.0;
	m4(1, 2) = 9;
	Matrix m5 = m3 + m4;
	Matrix m6 = m3 * m4;
	cout << m5 << endl;
	cout << m6 << endl;
#endif
	return 0;
}
