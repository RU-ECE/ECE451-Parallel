#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <thread>

using namespace std;
using namespace std::chrono;

extern "C" {
	//                     %rsi        %rdi
    void sequential(const char a[], uint64_t n);
    void backward(const char a[], uint64_t n);
	void skip2(const char a[], uint64_t n);
	void read1(const uint64_t* a, uint64_t n);
		void readfromreg(const uint64_t* a, uint64_t n);
}

int main() {
	const uint64_t n = 2LL*1024*1024*1024;
	char* a = (char*)aligned_alloc(n, 64); // 2GB
	
	//	sequential(a, n/16);

	/*
		On Dov's Aspire 3 15n Ryzen 5 with LPDDR5 RAM
		read1 = 1.4seconds with n=2G
		readfromreg = .7s with n=2G

	 */
	//  read1((uint64_t*)a, n);  
	//	readfromreg((uint64_t*)a, n);
	thread t1(read1, (uint64_t*)a, n);
	thread t2(read1, (uint64_t*)a, n);
	thread t3(read1, (uint64_t*)a, n);
	thread t4(read1, (uint64_t*)a, n);
  t1.join();
	t2.join();
	t3.join();
	t4.join();	
}
