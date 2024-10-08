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

void k(volatile double x[], int stride,int n) {
    const int size = 1024;
    for (int i = 0; i < n; i+=size) {
        for (int j = 0; j < size; j++)
            x[j]++;
    }
}

int main() {
    const int n = 100'000'000;
    const int num_trials = 100;
    double* x = new double[n];

    double sum = 0;
    auto start = chrono::high_resolution_clock::now();
    for (int trial = 0; trial < num_trials; trial++) {
        sum  = a(x, n);
    }
    auto end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6/num_trials << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    for (int trial = 0; trial < num_trials; trial++) {
        sum  = b(x, n);
    }
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6/num_trials << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    for (int trial = 0; trial < num_trials; trial++) {
        sum  = c(x, n);
    }
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6/num_trials << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    for (int trial = 0; trial < 10; trial++) {
        sum  = d(x, n);
    }
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6/num_trials << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    for (int trial = 0; trial < num_trials; trial++) {
        e(x, n);
    }
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6/num_trials << "sum = " << sum << endl;

    for (int stride = 1; stride <= 1024; stride*=2) {
        start = chrono::high_resolution_clock::now();
        for (int trial = 0; trial < num_trials; trial++) {
            f(x, stride, n);
        }
        end = chrono::high_resolution_clock::now();
        cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6/num_trials << "sum = " << sum << endl;
    }
    start = chrono::high_resolution_clock::now();
    for (int trial = 0; trial < num_trials; trial++) {
        sum  = g(x, 16, n);
    }
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6/num_trials << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    for (int trial = 0; trial < num_trials; trial++) {
        sum  = h(x, 16, n);
    }
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6/num_trials << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    for (int trial = 0; trial < 10; trial++) {
        sum  = j(x, 16, n);
    }
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6/num_trials << "sum = " << sum << endl;

    start = chrono::high_resolution_clock::now();
    for (int trial = 0; trial < num_trials; trial++) {
        k(x, 16, n);
    }
    end = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::microseconds>(end - start).count()*1e-6/10 << "sum = " << sum << endl;
    
    return 0;
}
