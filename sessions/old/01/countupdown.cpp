#include <iostream>
using namespace std;

int g(int x);
int f(int x);

auto count = 0; // initialize global variable to zero (good style)
constexpr auto n = 1'000'000'000;
void increment() {
    for (auto i = 0; i < n; i++) {
        count = f(count);
    }
}

void decrement() {
    for (auto i = 0; i < n; i++) {
        count = g(count);
    }
}

int main() {
    increment();
    decrement();
    cout << count << endl;
    return 0;
}