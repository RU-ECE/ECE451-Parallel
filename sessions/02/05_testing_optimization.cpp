#include <iostream>

using namespace std;

int main() {
	constexpr auto n = 1'000'000'000;
	auto sum = 0UL;
	for (auto i = 1; i <= n; i++)
		sum += i;
	cout << "sum=" << sum << endl;
}
