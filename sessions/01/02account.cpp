#include <iostream>
#include <thread>
#include <unistd.h>

using namespace std;

auto balance = 0;

void deposit(const int count, const int amount) {
	for (auto i = 0; i < count; i++) {
		// balance += amount;
		const auto tmp = balance + amount;
		cout << "deposit: " << tmp << " amt=" << amount << endl;
		sleep(1);
		balance = tmp;
	}
	// cout << "Deposit: " << balance << endl;
}
void withdraw(const int count, const int amount) {
	for (auto i = 0; i < count; i++) {
		if (balance < amount)
			return;
		const int tmp = balance - amount;
		cout << "withdraw: " << balance << " amt=" << amount << endl;
		sleep(1);
		balance = tmp;
	}
	// cout << "Withdraw: " << balance << endl;
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
