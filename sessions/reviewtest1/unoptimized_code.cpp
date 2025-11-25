#include <cstdint>
#include <iostream>

using namespace std;

uint64_t f(const uint64_t x, const uint64_t y) { return x + y; }

int main() {
	constexpr uint64_t x = 3;
	constexpr uint64_t y = 5;
	cout << f(x, y);
	return 0;
}
