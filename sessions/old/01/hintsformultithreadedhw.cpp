#include <thread>
#include <iostream>

using namespace std;

constexpr auto num_threads = 16;
int allcount[num_threads]; // array of counts for each thread

void f(int min, int max,int* globalcount) {
	auto mycount = 0;
    for (auto i = 0; i < 1000; i++) {
        mycount++;
    }
    *globalcount += mycount;
}

void testkthreads(const int k) {
	auto globalcount = 0;
    thread* t[k];
    for (auto i = 0; i < k; i++) {
        t[i] = new thread(f, 0, 50, &globalcount);
    }
    for (auto i = 0; i < k; i++) {
        t[i]->join(); // wait for all threads to finish
    }
    cout << globalcount << endl;
} 
int main() {
    for (auto k = 1; k <= 16; k *= 2) {
		const clock_t t0 = clock();
        testkthreads(k);
		const clock_t t1 = clock();
        cout << (double)(t1 - t0) / CLOCKS_PER_SEC << endl;
    }
}