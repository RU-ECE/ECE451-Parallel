#include <iostream>
#include <thread>
#include <unistd.h>
#include <mutex>

using namespace std;

int balance =0;
mutex m;

void deposit(int amount) {
    m.lock();
    balance += amount;
    m.unlock();
}

bool withdraw(int amount) {
    if (balance >= amount) {
        m.lock();
        balance -= amount;
        m.unlock();
        return true;       
    }
    return false;
}

void processDeposits(int n) {
    for (int i = 0; i < n; i++) {
        deposit(1);
    }
}

void processWithdrawals(int n) {
    for (int i = 0; i < n; i++) {
        withdraw(1);
    }
}

int main() {
#if 0
    processDeposits(10000);
    processWithdrawals(10000);
#endif

    const int n = 100'000'000;
    thread t1(processDeposits, n);
    usleep(10000);
    thread t2(processWithdrawals, n);
    t1.join();
    t2.join();
    cout << balance << endl;
    return 0;

}
