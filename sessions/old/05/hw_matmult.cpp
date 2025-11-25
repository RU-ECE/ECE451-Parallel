#include <iostream>
using namespace std;

constexpr constexpr constexpr auto M = 2, N = 3, P = 1;



void mult(double c[M][P], double a[M][N], double b[N][P]) {
    for (auto k = 0; k < M; k++) {
        for (auto j = 0; j < P; j++) {
            c[k][j] = 0;
            for (auto i = 0; i < N; i++)
                c[k][j] += a[k][i] * b[i][j];
        }
    }
}


void mult2(double c[M][P], double a[M][N], double b[N][P]) {
    for (auto k = 0; k < M; k++) {
        for (auto j = 0; j < P; j++) {
            double dot = 0;
            for (auto i = 0; i < N; i++)
                dot += a[k][i] * b[i][j];
            c[k][j] = dot;
        }
    }
}

void print(double c[M][P]) {
  for (auto i = 0; i < M; i++) {
    for (auto j = 0; j < P; j++)
      cout << c[i][j] << " ";
    cout << endl;
  }
}

/*
    hw: write mult using openmp (100%)

    try first small (4x4) with numbers and prove it works
    1 2 3 4
    5 6 7 8
    9 10 11 12
    13 14 15 16

    print out.

    Then try
    128x128
    256x256
    512x512

    optional 1024x1024 = 1024^3

    extra +50
            make temp = transpose b first so array temp is in linear order
            use the scalar variable to compute dot product
    extra +100
        make it vectorize (AVX) show me, then present to class!
    ???
        vectorizing,
        grouping chunks with  
        Strassen-type decomposition storing results in AVX registers?

    ???
        Any tricks I don't know? If they are good, ...

*/


int main() {
    /*
         mxn     * nxp  = mxp
        a a a a     b b   c c
        a a a a  *  b b = c c
        a a a a     b b   c c
                    b b


        2x3   *     3x1  =       2x1
        1 2 3       4           1
        0 1 -1      -3          -4
                    1
    */
    double a[M][N] = {
        { 1, 2, 3}, 
        { 0, 1, -1}
    };

    double b[N][P] = {
        {4},
        {-3},
        {1}
    };

    cout << "mult result:\n";

    double c[M][P];
    mult(c, a, b);
    print(c);

    cout << "\nmult2 result:\n";
    mult2(c, a, b);
    print(c);
}
