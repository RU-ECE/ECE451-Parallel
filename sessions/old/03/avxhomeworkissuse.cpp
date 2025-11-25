#include <cmath>
/*
	how to return multiple values in registers in C++??


*/
using namespace std;

void rect2polar(const float x, const float y, float& r2, float& theta2) {
    r2 = sqrt(x * x + y * y);
    theta2 = atan2(y, x);
}


struct Polar {
    float r, theta;
};

Polar rect2polar(const float x, const float y) {
    return {sqrt(x * x + y * y), atan2(y, x)};
}



int main() {
    float x = 3, y = 4;
    float r, theta;

    rect2polar(x, y, r, theta);

}