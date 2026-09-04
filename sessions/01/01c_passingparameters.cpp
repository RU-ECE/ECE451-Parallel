#include <iostream>
#include <thread>

using namespace std;

// n is the total number we are doing
int globalvar[10]; // pass in any information you want...


uint64_t counts[4];

void countprime(uint64_t n, uint64_t startat, uint64_t size, uint64_t step) {
   // do your prime number stuff
   startat = startat + step;

}

void f(uint64_t a, uint64_t b) {
    cout << "entering thread f:" << a << "," << b << '\n';
}


int main() {
    const int n= 100'000'000;
    const int batchsize = 1'000'000;
    const int step = batchsize*2;
    thread t1(countprime, n, 2, batchsize, step, &counts[0]);
    thread t2(f, batchsize+1, 2*batchsize, step, &counts[1]);
    t1.join();
    uint64_t total_primes = counts[0] + counts[1];
    // at the end , all threads add up their counts to get the total

}