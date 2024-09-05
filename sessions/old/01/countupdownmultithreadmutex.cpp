#include <iostream>
#include <thread>
#include <mutex>
using namespace std;

mutex m;

/*
    race condition

*/

int g(int x);
int f(int x);

int count = 0; // initialize global variable to zero (good style)
const int n = 1'000'000'0;
void increment() {
    for (int i = 0; i < n; i++) {
        lock_guard<mutex> lock(m);
        count = f(count);
    }
}

void decrement() {
    for (int i = 0; i < n; i++) {
        lock_guard<mutex> lock(m);
        count = g(count);
    }
}

int main() {
    thread t1(increment);
    thread t2(decrement);
    t1.join();
    t2.join();

    cout << count << endl;
    return 0;
}