#include <iostream>
#include <mutex>
#include <thread>

using namespace std;

auto balance = 0;
mutex m;

void deposit(const int count, const int amount) {
	for (auto i = 0; i < count; i++) {
		m.lock();
		balance += amount;
		m.unlock();
		//        cout << "deposit: " << balance << " amt=" << amount << endl;
	}
	//    cout << "Deposit: " << balance << endl;
}
void withdraw(const int count, const int amount) {
	for (auto i = 0; i < count; i++) {
		m.lock();
		balance -= amount;
		m.unlock();
		//        cout << "withdraw: " << balance << " amt=" << amount << endl;
	}
}
int main() {
	constexpr auto n = 1'000'000;
	thread t1(deposit, n, 1);
	thread t2(withdraw, n, 1);
	t1.join();
	t2.join();
	cout << "Balance: " << balance << endl;
	return 0;
}
