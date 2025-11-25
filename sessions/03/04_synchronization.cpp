#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>

using namespace std;

auto balance = 0;
mutex m;

void deposit(const int amount) {
	m.lock();
	balance += amount;
	m.unlock();
}

bool withdraw(const int amount) {
	if (balance >= amount) {
		m.lock();
		balance -= amount;
		m.unlock();
		return true;
	}
	return false;
}

void processDeposits(const int n) {
	for (auto i = 0; i < n; i++)
		deposit(1);
}

void processWithdrawals(const int n) {
	for (auto i = 0; i < n; i++)
		withdraw(1);
}

int main() {
#if 0
    processDeposits(10000);
    processWithdrawals(10000);
#endif

	constexpr auto n = 100'000'000;
	thread t1(processDeposits, n);
	usleep(10000);
	thread t2(processWithdrawals, n);
	t1.join();
	t2.join();
	cout << balance << endl;
	return 0;
}
