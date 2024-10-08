#include <iostream>
#include <chrono>
using namespace std;

double a(volatile double x[], int n) {
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += x[i];
    return sum;
}

double b(volatile double x[], int n) {
    double sum = 0;
    for (int i = n-1; i >= 0; i--)
    sum += x[i];
    return sum;
}

double c(volatile double x[], int n) {
    double sum = 0;
    for (int i = 0; i < n; i+=16)
        sum += x[i];
    return sum;
}

double d(volatile double x[], int n) {
    double sum = 0;
    for (int i = 0; i < n; i+=16)
        sum += x[i];
    return sum;
}

void e(volatile double x[], int n) {
    for (int i = 0; i < n; i+=16)
       x[i]++;
}

void f(volatile double x[], int stride, int n) {
    for (int i = 0; i < stride; i++) {
        for (int j = i; i < n; i+=stride)
            x[j]++;
    }
}

double g(volatile double x[], int stride,int n) {
    double sum = 0;
    for (int i = 0; i < stride; i++) {
        for (int j = i; j < n; j+=stride)
            sum += x[j];
    }
    return sum;
}

double h(volatile double x[], int stride,int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[0];
    }
    return sum;
}

double j(volatile double x[], int stride,int n) {
    const int size = 1024;
    double sum = 0;
    for (int i = 0; i < n; i+=size) {
        for (int j = 0; j < size; j++)
            sum += x[j];
    }
    return sum;
}

int main() {
    const int n = 100'000'000;
    double* x = new double[n];

    double sum;
    auto start = chrono::high_resolution_clock::now();
    sum  = a(x, n);
    auto end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6 << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    sum  = b(x, n);
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6 << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    sum  = c(x, n);
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6 << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    sum  = d(x, n);
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6 << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    e(x, n);
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6 << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    sum  = f(x, n);
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6 << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    sum  = g(x, 16, n);
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6 << "sum = " << sum << endl;

    return 0;
}
