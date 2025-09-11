#include <iostream>
using namespace std;

int main() {
    const int n = 1'000'000'000;
    uint64_t sum = 0;
    for (int i = 1; i <= n; i++) {
        sum += i;
    }
    cout << "sum="<< sum << '\n';

}