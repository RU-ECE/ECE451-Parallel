#include <immintrin.h>
#include <iostream>

using namespace std;

double sum(const double x[], int n) {
    __m256d sum = _mm256_setzero_pd();
    for (int i = 0; i < n; i += 4) {
        __m256d v = _mm256_load_pd(&x[i]);
        sum = _mm256_add_pd(sum, v);
    }
    return sum[0] + sum[1] + sum[2] + sum[3];
}
    
int main() {
    double x[] = {1, 2, 3, 4, 5, 6, 7, 8};
    cout << sum(x, 8) << endl;

    return 0;
}