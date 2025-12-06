#include <iostream>

using namespace std;

unsigned long f(const unsigned long x, const unsigned long y) { return x + y; }

int main() {
	constexpr auto x = 3UL, y = 5UL;
	cout << f(x, y);
	return 0;
}
