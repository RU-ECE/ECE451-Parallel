#include <thread>
#include <iostream>
#include <unistd.h>
#include <mutex>
using namespace std;

int balance = 0;
mutex m;

void deposit(int count, int amount) {
    for (int i = 0; i < count; i++) {
        m.lock();
        balance += amount;
        m.unlock();
//        cout << "deposit: " << balance << " amt=" << amount << endl;
    }
//    cout << "Deposit: " << balance << endl;
}
void withdraw(int count, int amount) {
    for (int i = 0; i < count; i++) {
        m.lock();
        balance -= amount;
        m.unlock();
//        cout << "withdraw: " << balance << " amt=" << amount << endl;
    }
}
int main() {
    const int n = 1'000'000;
    thread t1(deposit, n, 1);
    thread t2(withdraw, n, 1);
    t1.join();
    t2.join();
    cout << "Balance: " << balance << endl;
    return 0;
}