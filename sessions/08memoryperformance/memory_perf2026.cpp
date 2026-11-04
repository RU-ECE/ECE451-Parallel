#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstdint>

using namespace std;
using namespace std::chrono;

extern "C" {
	//                     %rsi        %rdi
    void sequential(const char a[], uint64_t n);
    void backward(const char a[], uint64_t n);
	void skip2(const char a[], uint64_t n);
}

int main() {
	const uint64_t n = 2LL*1024*1024*1024;
	char* a = (char*)aligned_alloc(n, 64); // 2GB
	
	sequential(a, n/16);

}
