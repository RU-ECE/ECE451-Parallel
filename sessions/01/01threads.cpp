#include <iostream>
#include <cmath>
#include <thread>
#include <stdexcept>
#include <unistd.h>
#include <chrono>
using namespace std;

//28 = 1 ,2 , 4    |   7,    14,  28


// O(sqrt(n))  omega(1)    is n prime? n=1001    n%2, n%3,  .. n%33 n%500
// n=28    1, 2,4    |   7,    14,  28
bool isPrime(uint64_t n) {
    for (uint64_t i = 2; i <= sqrt(n); i++) {
      if (n % i == 0)
        return false;
    }
    return true;
}

void f() {
    for (;;) { // infinite loop
			cout << "hello" << flush;
        usleep(100000);
    }
}

void g() {
    while (true) { 
//        int* p = new int[1024L*1024*1024 * 16];
			auto t0 = std::chrono::steady_clock::now();			
			cout << "bye" << flush;
			auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now() - t0
).count();
			cout << "elapsed: " << elapsed << flush;
        usleep(200000);
    }
}

int main() {
   try {
    thread t1(f);
    thread t2(g);
//    t1.join();
//    t2.join();
  } catch (exception e) {
     cout << e.what() << '\n';
  }
}




