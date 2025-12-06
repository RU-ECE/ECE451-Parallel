#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

using namespace std;
using namespace chrono;

// learning how the compiler maps c++ into optimal assembly language: 64-bit int math
unsigned long uint64_math(const unsigned long a, const unsigned long b) {
	const auto p = a + b, q = a - b, r = a * b, s = a / b, w = a % b, x = -a;
	return p * q - r * s - w - x * x; // TODO: is the compiler going to optimize -a out?
	// because (-a)*(-a) is a*a
}

// learning how the compiler maps c++ into optimal assembly language: 64-bit int math
float float_math(const float a, const float b) {
	const auto p = a + b;
	const auto q = a - b;
	const auto r = a * b;
	const auto s = a / b;
	// float w = a % b;  // no mod in floating point, there is fmod(x,y) = x - y * floor(x/y)
	const auto x = -a;
	const auto y = abs(a);
	const auto z = sqrt(a);
	return p * q - r * s - 3 * x + y + z; // note: probably won't optimize 3*x+y out
	// order of floating point changes results, and compiler writers are afraid to
	// optimize like this. Turning on unsafe optimizations -O3? or a specific flag will
	// eventually do some, but not necessarily all.
}

// learning how the compiler maps c++ into optimal assembly language: 64-bit int math
double double_math(const double a, const double b) {
	const auto p = a + b;
	const auto q = a - b;
	const auto r = a * b;
	const auto s = a / b;
	// double w = a % b; // no mod in floating point, there is fmod
	const auto x = -a;
	const auto y = abs(a);
	const auto z = sqrt(a);
	return p * q - r * s - 3 * x + y + z; // TODO: is the compiler going to optimize 3*x+y out?
}

// this was failing to optimize, but now seems to work, *sigh*
// something was stopping optimization from working in-register
// can't duplicate issue. It's in the original code though
inline bool isPrime(const unsigned long n) {
	const auto lim = static_cast<unsigned long>(sqrt(n));
	for (auto i = 2UL; i <= lim; i++)
		if (n % i == 0) // if n is evenly divisible by i
			return false;
	return true;
}

// do the counting in a register, do not write to memory
void countPrimesMultithreaded(unsigned long a, const unsigned long b, unsigned long* pcount) {
	cout << "countPrimesMultithreaded: " << a << "," << b << endl;
	auto count = a == 2 ? 1UL : 0UL;
	a |= 1; // 10101010101010101010101010100001
	// THE ABOVE GUARANTEES THAT a IS odd
	for (auto i = a; i <= b; i += 2) // O(n)
		if (isPrime(i)) // O(sqrt(n))
			count++;
	*pcount = count;
}

// do the counting in a register, do not write to memory
void countPrimesMultithreaded2(unsigned long a, const unsigned long b, unsigned long* pcount) {
	cout << "countPrimesMultithreaded2: " << a << "," << b << endl;
	auto count = a == 2 ? 1UL : 0UL;
	a |= 1; // 10101010101010101010101010100001
	// round up to next odd number using OR
	// THE ABOVE GUARANTEES THAT a IS odd
	for (auto i = a; i <= b; i += 2) { // O(n)
		const auto lim = static_cast<unsigned long>(sqrt(i));
		auto j = 0UL;
		for (j = 3; j <= lim && i % j != 0; j += 2) // O(sqrt(n))
			;
		if (j > lim)
			count++;
	}
	*pcount = count;
}

int main(int, char* argv[]) {
	const auto r = uint64_math(1, 2); // this will probably be optimized out at compile time
	cout << r << endl; // must use r or optimizer will definitely destroy all the code as dead
	{
		// just to demonstrate, this generates NO CODE because we don't use r
		float x = float_math(1, 2);
		// however, the compiler does generate code for float_math because it doesn't know
		// whether some other part of the program might call it
		// in C++, the linker is not part of compilation, unlike Java.
		// so we can look at the assembler for the function itself
	}
	const auto n = strtol(argv[1], nullptr, 10); // count primes up to 1 million single threaded
	{
		cout << "\n=========\n1 thread writing to memory:\n";
		unsigned long count;
		const auto t0 = high_resolution_clock::now();
		countPrimesMultithreaded(2, n, &count);
		const auto t1 = high_resolution_clock::now();
		cout << "prime count=" << count << " elapsed: " << duration_cast<microseconds>(t1 - t0).count() * 1e-6 << endl;
	}
	{
		cout << "\n=========\n1 thread in registers:\n";
		auto count = 0UL;
		const auto t0 = high_resolution_clock::now();
		countPrimesMultithreaded2(2, n, &count);
		const auto t1 = high_resolution_clock::now();
		cout << "prime count=" << count << " elapsed: " << duration_cast<microseconds>(t1 - t0).count() * 1e-6 << endl;
	}
	{
		cout << "\n=========\n2 threads:\n";
		auto count1 = 0UL, count2 = 0UL;
		const auto start = high_resolution_clock::now();
		thread t1(countPrimesMultithreaded2, 2, n / 2, &count1);
		thread t2(countPrimesMultithreaded2, n / 2 + 1, n, &count2);
		t1.join();
		t2.join();
		const auto count = count1 + count2;
		const auto end = high_resolution_clock::now();
		cout << "prime count=" << count << " elapsed: " << duration_cast<microseconds>(end - start).count() * 1e-6
			 << endl;
	}
	{
		cout << "\n=========\n4 threads:\n";
		auto count1 = 0UL, count2 = 0UL, count3 = 0UL, count4 = 0UL;
		const auto start = high_resolution_clock::now();
		thread t1(countPrimesMultithreaded2, 2, n / 4, &count1);
		thread t2(countPrimesMultithreaded2, n / 4 + 1, n / 2, &count2);
		thread t3(countPrimesMultithreaded2, n / 2 + 1, 3 * n / 4, &count3);
		thread t4(countPrimesMultithreaded2, 3 * n / 4 + 1, n, &count4);
		t1.join();
		t2.join();
		t3.join();
		t4.join();
		const auto count = count1 + count2 + count3 + count4;
		const auto end = high_resolution_clock::now();
		cout << "prime count=" << count << " elapsed: " << duration_cast<microseconds>(end - start).count() * 1e-6
			 << endl;
	}
	return 0;
}
