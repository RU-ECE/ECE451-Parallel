#include <iostream>
#include <thread>
using namespace std;

int g(int x);
int f(int x);

int count = 0; // initialize global variable to zero (good style)
const int n = 1'000'000'000;
void increment() {
    for (int i = 0; i < n; i++) {
        count = f(count);
    }
}

void decrement() {
    for (int i = 0; i < n; i++) {
        count = g(count);
    }
}

int main() {
    increment();
    decrement();
    cout << count << endl;
    return 0;
}