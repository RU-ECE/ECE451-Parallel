#include <iostream>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <thread>

using namespace std;

// learning how the compiler maps c++ into optimal assembly language: 64-bit int math
uint64_t uint64_math(uint64_t a, uint64_t b) {
    uint64_t p = a + b;
    uint64_t q = a - b;
    uint64_t r = a * b;
    uint64_t s = a / b;
    uint64_t w = a % b;
    uint64_t x = -a;
    return p * q - r * s - w - x*x; //TODO: is the compiler going to optimize -a out?
    // because (-a)*(-a) is a*a
}

// learning how the compiler maps c++ into optimal assembly language: 64-bit int math
float float_math(float a, float b) {
    float p = a + b;
    float q = a - b;
    float r = a * b;
    float s = a / b;
    // float w = a % b;  // no mod in floating point, there is fmod(x,y) = x - y * floor(x/y)
    float x = -a;
    float y = abs(a);
    float z = sqrt(a);
    return p * q - r * s - 3*x + y + z; //note: probably won't optimize 3*x+y out
    // order of floating point changes results, and compiler writers are afraid to
    // optimize like this. Turning on unsafe optimizations -O3? or a specific flag will
    // eventually do some, but not necessarily all.
}

// learning how the compiler maps c++ into optimal assembly language: 64-bit int math
double double_math(double a, double b) {
    double p = a + b;
    double q = a - b;
    double r = a * b;
    double s = a / b;
//    double w = a % b; // no mod in floating point, there is fmod
    double x = -a;
    double y = abs(a);
    double z = sqrt(a);
    return p * q - r * s - 3*x + y + z; //TODO: is the compiler going to optimize 3*x+y out?
}

inline bool isPrime(uint64_t n) {
    uint64_t lim = sqrt(n);
    for (uint64_t i = 2; i <= lim; i++) {
      if (n % i == 0) // if n is evenly divisible by i
        return false;
    }
    return true;
}
// do the counting in a register, do not write to memory
void countPrimesMultithreaded(uint64_t a, uint64_t b, uint64_t* pcount) {
    uint64_t count = (a == 2 ? 1 : 0);
    a |= 1; // 10101010101010101010101010100001
  // THE ABOVE GUARANTEES THAT a IS odd 
    for (int i = a; i <= b; i+=2) //O(n)
      if (isPrime(i)) //O(sqrt(n))
        count++;
    *pcount = count;
}

// do the counting in a register, do not write to memory
void countPrimesMultithreaded2(uint64_t a, uint64_t b, uint64_t* pcount) {
    uint64_t count = a == 2 ? 1 : 0;
    a |= 1; // 10101010101010101010101010100001
// round up to next odd number using OR
  // THE ABOVE GUARANTEES THAT a IS odd 
    for (int i = a; i <= b; i+=2) { //O(n)
      const uint64_t lim = sqrt(i);
      int j;
      for (j = 3; j <= lim && i % j != 0; j+=2) //O(sqrt(n))
        ;
      if (j > lim)
        count++;
    }
    *pcount = count;
}


int main(int argc, char* argv[]) {
    uint64_t r = uint64_math(1,2); // this will probably be optimized out at compile time
    cout << r << endl; // must use r or optimizer will definitely destroy all the code as dead

    {
        // just to demonstrate, this generates NO CODE because we don't use r
        float r = float_math(1,2);
        // however, the compiler does generate code for float_math because it doesn't know
        // whether some other part of the program might call it
        // in C++, the linker is not part of compilation, unlike Java.
        // so we can look at the assembler for the function itself
    }
    const int n = atoi(argv[1]); // count primes up to 1 million single threaded
    {
        uint64_t count;
        auto t0 = chrono::high_resolution_clock::now();
        countPrimesMultithreaded(2, n, &count);
        auto t1 = chrono::high_resolution_clock::now();
        cout << "prime count=" << count << " elapsed: " << chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-6 << endl;
    }
    {
        uint64_t count = 0;
        auto t0 = chrono::high_resolution_clock::now();
        countPrimesMultithreaded2(2, n, &count);
        auto t1 = chrono::high_resolution_clock::now();
        cout << "prime count=" << count << " elapsed: " << chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()*1e-6 << endl;
    }
    {
        uint64_t count1 = 0, count2 = 0;
        auto start = chrono::high_resolution_clock::now();
        thread t1(countPrimesMultithreaded2, 2, n/2, &count1);
        thread t2(countPrimesMultithreaded2, 2, n/2, &count2);
        t1.join();
        t2.join();
        uint64_t count = count1 + count2;
        auto end = chrono::high_resolution_clock::now();
        cout << "prime count=" << count << " elapsed: " << chrono::duration_cast<std::chrono::microseconds>(end - start).count()*1e-6 << endl;
    }
    return 0;
}