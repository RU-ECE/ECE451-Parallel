#include <cstdint>
#include <iostream>

using namespace std;

int main() {
	constexpr auto n = 1'000'000'000;
	uint64_t sum = 0;
	for (auto i = 1; i <= n; i++)
		sum += i;
	cout << "sum=" << sum << endl;
}
