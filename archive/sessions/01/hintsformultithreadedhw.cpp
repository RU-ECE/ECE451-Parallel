#include <iostream>
#include <thread>

using namespace std;

constexpr auto num_threads = 16;
int allCount[num_threads]; // array of counts for each thread

void f(int min, int max, int* globalCount) {
	auto myCount = 0;
	for (auto i = 0; i < 1000; i++)
		myCount++;
	*globalCount += myCount;
}

void testkthreads(const int k) {
	auto globalCount = 0;
	thread* t[k];
	for (auto i = 0; i < k; i++)
		t[i] = new thread(f, 0, 50, &globalCount);
	for (auto i = 0; i < k; i++)
		t[i]->join(); // wait for all threads to finish
	cout << globalCount << endl;
}

int main() {
	for (auto k = 1; k <= 16; k *= 2) {
		const auto t0 = clock();
		testkthreads(k);
		const auto t1 = clock();
		cout << static_cast<double>(t1 - t0) / CLOCKS_PER_SEC << endl;
	}
}
