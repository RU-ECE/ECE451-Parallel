#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
using namespace std;

// example n = 1000000001
// O(sqrt n)  omega(1)
bool isPrime(uint64_t n) {
    uint64_t lim = sqrt(n);
    if (n % 2 == 0)
      return false;
    for (uint64_t i = 3; i <= lim; i+=2) {
      if (n % i == 0) // if n is evenly divisible by i
        return false;
    }
    return true;
}

uint64_t countPrimes(uint64_t n) { //O(n sqrt(n))
    uint64_t count = 1;
    for (int i = 3; i <= n; i+=2) //O(n)
      if (isPrime(i)) //O(sqrt(n))
        count++;
    return count;
}
// n = 10^9
// 2-1K, 1K+1-2K, 2K+1-3K, 

void countPrimesMultithreaded(uint64_t a, uint64_t b, uint64_t* pcount) {
//    uint64_t count = 1;
    *pcount = (a == 2 ? 1 : 0);
    a |= 1; // 10101010101010101010101010100001
//    if (a % 2 == 0)
//      a++;
    for (int i = a; i <= b; i+=2) //O(n)
      if (isPrime(i)) //O(sqrt(n))
        (*pcount)++;
}

int main(int argc, char* argv[]) {
    uint64_t n = atol(argv[1]);
    uint64_t chunkSize = 1024*1024;
    uint64_t current = 2;
//    std::cout << countPrimes(n) << '\n';
    uint64_t count1 = 0, count2 = 0;
    thread t1(countPrimesMultithreaded, 2, n/2, &count1);
    thread t2(countPrimesMultithreaded, n/2+1, n, &count2);
    t1.join();
    t2.join();
    uint64_t count = count1 + count2;
    cout << count << '\n';

}