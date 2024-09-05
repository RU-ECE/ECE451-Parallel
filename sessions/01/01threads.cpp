#include <iostream>
#include <cmath>
#include <thread>
#include <stdexcept>
#include <unistd.h>
using namespace std;

//28 = 1 ,2 , 4    |   7,    14,  28


// O(sqrt(n))  omega(1)
bool isPrime(uint64_t n) {
    for (uint64_t i = 2; i <= sqrt(n); i++) {
      if (n % i == 0)
        return false;
    }
    return true;
}

void f() {
    for (;;) {
        cout << "hello";
        usleep(1000000);
    }
}

void g() {
    while (true) { 
//        int* p = new int[1024L*1024*1024 * 16];
        cout << "bye" << flush;
        usleep(2000000);
    }
}

int main() {
   try {
    thread t2(g);
    thread t1(f);
    t1.join();
    t2.join();
  } catch (exception e) {
     cout << e.what() << '\n';
  }
}




