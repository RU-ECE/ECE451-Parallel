#include <iostream>

using namespace std;
class Matrix {
private:
    uint32_t rows;
    uint32_t cols;
    double *data;
    Matrix(uint32_t r, uint32_t c) : rows(r), cols(c), data(new double[rows*cols]) {}
public:
    Matrix(uint32_t r, uint32_t c, double v) : rows(r), cols(c), data(new double[rows*cols]) { // initializer list
        for (int i = 0; i < rows*cols; i++) {
            data[i] = v;
        }
    }

    ~Matrix() {
        delete[] data; // it is implementation-defined what happens if you don't delete []
    }

// copy constructor only happens to uninitialized memory
    Matrix(const Matrix& orig) : rows(orig.rows), cols(orig.cols), data(new double[rows*cols]) {
        for (int i = 0; i < rows*cols; i++) {
            data[i] = orig.data[i];
        }
    }

//operator = must first give back the memory you already have, then copy
    Matrix& operator =(Matrix copy) {
        rows = copy.rows;
        cols = copy.cols;
        swap(data, copy.data);
        return *this;
    }
    Matrix(Matrix&& orig) : rows(orig.rows), cols(orig.cols), data(orig.data) {
        orig.data = nullptr; // rob the dying object of its memory
    }

    friend Matrix operator +(const Matrix& a, const Matrix& b) {
        if (a.rows != b.rows || a.cols != b.cols) {
            throw std::invalid_argument("Matrix sizes don't match");
        }
        Matrix ans(a.rows, a.cols, 0.0); // this is inefficient, creates matrix of all zeros
        for (int i = 0; i < a.rows*a.cols; i++) {
            ans.data[i] = a.data[i] + b.data[i];
        }
        return ans;
    }

    double operator ()(int r, int c) const { // this is READONLY
        return data[r*cols + c];
    }
    double& operator ()(int r, int c) { // this operator can change m(r,c)
        return data[r*cols + c];
    }

    friend Matrix operator *(const Matrix& a, const Matrix& b) {
        if (a.cols != b.rows) {
            throw std::invalid_argument("Matrix sizes don't match");
        }
        Matrix ans(a.rows, b.cols);
        for (int i = 0; i < a.rows; i++) {
            double dot = 0;
            for (int j = 0; j < b.cols; j++) {
                for (int k = 0; k < a.cols; k++) {
                    dot += a(i,k) * b(k,j);
                }
                ans(i,j) = dot;
            }
        }
        return ans;
    }

    friend ostream& operator <<(ostream& os, const Matrix& m) {
        for (int i = 0, c = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++, c++) {
                os << m.data[c] << " ";
            }
            os << '\n';
        }
        return os;
    }

    static Matrix ident(uint32_t n) {
        Matrix ans(n, n, 0.0);
        for (int i = 0; i < n; i++) {
            ans(i,i) = 1.0;
        }
        return ans;
    }
};
int main() {
    Matrix m1(4, 3, 0.0);
    Matrix m2(4, 3, 0.0);
    cout << m1(2,3) << endl;
    m1(2,3) = 2.0;
    m2(3,3) = -2.0;
    Matrix m3 = Matrix::ident(4);
    Matrix m4 = Matrix::ident(4);
    m4(3,3) = -2.0;
    m4(1,2) = 9;
    Matrix m5 = m1 + m2;
    Matrix m6 = m3 * m4;
    cout << m5 << endl;
    cout << m6 << endl;
    return 0;
}