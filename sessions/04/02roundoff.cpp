#include <iostream>
#include <iomanip>
#include <cmath>
// 100'000'000
int main() {
    for (float f = 1e8; f <= 1e8+1; f += 1.0) {
        std::cout << std::setprecision(15) << f << std::endl;
    }
    return 0;
 }