#include <iostream>
using namespace std;

uint64_t f(uint64_t x, uint64_t y) {
    return x + y;
}

int main() {
    uint64_t x = 3;
    uint64_t y = 5;
    cout << f(x,y);
    return 0;
}