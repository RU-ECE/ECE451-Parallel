#include <thread>
#include <iostream>
#include <unistd.h>
using namespace std;

int balance = 0;

void deposit(int count, int amount) {
    for (int i = 0; i < count; i++) {
        //balance += amount;
        int tmp = balance + amount;
        cout << "deposit: " << tmp << " amt=" << amount << endl;
        sleep(1);
        balance = tmp;
    }
//    cout << "Deposit: " << balance << endl;
}
void withdraw(int count, int amount) {
    for (int i = 0; i < count; i++) {
        if (balance < amount)
            return;
        int tmp = balance - amount;
        cout << "withdraw: " << balance << " amt=" << amount << endl;
        sleep(1);
        balance = tmp;
    }
//    cout << "Withdraw: " << balance << endl;
}
int main() {
    thread t1(deposit, 10, 1);
    sleep(1);
    thread t2(withdraw, 10, 1);
    t1.join();
    t2.join();
    cout << "Balance: " << balance << endl;
    return 0;
}