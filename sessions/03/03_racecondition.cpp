#include <iostream>
#include <thread>
#include <unistd.h>

using namespace std;

int balance =0;

void deposit(int amount) {
  //    sleep(1);
    int tmp = balance + amount;
    //    sleep(1);
    //cout << "depositing " << amount << endl;
    balance = tmp;
}

/*
race condition

    thread1: load balance:   rax = 0
    thread2: load balance:   rax = 0
    thread1: rax+1 = 1
    thread1: store balance = 1
    thread2: rax = 0 so i do nothing
    // balance = 1

    thread1: load balance:   rax = 1
    thread2: load balance:   rax = 1
    thread1: rax+1 = 2
    thread2: rax-1 = 0
    thread 1 write out : balance = 2
    thread2: write out: balance = 0

*/

bool withdraw(int amount) {
  //    sleep(1);
  //cout << "withdrawing " << amount << endl;
    if (balance >= amount) {
        int tmp = balance - amount;
	//        sleep(1);
        balance = tmp;
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
    // synchronous is faster!
//    processDeposits(n);
//    processWithdrawals(n);
    thread t1(processDeposits, n);
    thread t2(processWithdrawals, n);
    t1.join();
    t2.join();
    cout << balance << endl;
    return 0;

}
