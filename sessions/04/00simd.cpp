#include <iostream>
#include <cstdint>
#include <immintrin.h>
using namespace std;

int main() {
    uint64_t a = 0;
    float f = 1.2345678f; //32 bit float
    double d = 1.23456789012345e+308;
    __m256d b = {0, 0, 0, 0};
    __m256d c = {1, 2, 3.5, 4.2};    
    //__m256f d = {1.5f, 2.2f, 3.1f, 4.2f, 1.5f, 2.2f, 3.1f, 4.2f};
}