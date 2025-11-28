#include <iostream>
#include <thread>

using namespace std;

void test1() {
	thread t1(f);
	t1.join();
	thread t2(f);
	t2.join();
}

auto balance = 0;
void test2() {

	thread t1(deposit);
	thread t2(withdraw);
	t1.join();
	t2.join();
}

auto balance = 0;
void test2() {

	thread t1(deposit);
	thread t2(withdraw);
	cout << balance << endl;
}
