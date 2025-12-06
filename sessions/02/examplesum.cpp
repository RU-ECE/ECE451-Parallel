#include <iostream>

using namespace std;

// compute the sum 1/i^2 for i= 1 to n
int main(int, char** argv) {
	const auto n = static_cast<int>(strtol(argv[1], nullptr, 10));
	float s = 0;
	// compute 1/1^2 + 1/2^2 + 1/3^2 + ...
	for (auto i = 1; i <= n; i++)
		s += 1.0 / (i * i); // 1.0/1 + 1.0/4 + ..
	cout << s << endl;
	return 0;
}
