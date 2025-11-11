#include <iostream>

// compute the sum 1/i^2 for i= 1 to n
int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    float s= 0;
    // compute 1/1^2 + 1/2^2 + 1/3^2 + ...
    for (size_t i = 1; i <= n; i++) {
        s += 1.0/(i*i); // 1.0/1 + 1.0/4 + ..
    }
    std::cout << s << std::endl;
    return 0;

}