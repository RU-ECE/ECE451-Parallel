#include <cmath>
#include <iomanip>
#include <iostream>

using namespace std;

/*
	1.23
	 .0887
	 .0925
	=======
	 .0887
+    .0925
==========
	 .1812
	 .181
	1.23
	=====
	1.41


	1.23
	 .0887
	 =====
	1.31
	 .09
	====
	1.40
*/

float euler1(const unsigned long n) {
	auto sum = 0.0f; // 1/1 + 1/4 + 1/9 + 1/16
	for (auto i = 1UL; i <= n; i++)
		sum += 1.0f / (i * i);
	return sum;
}

float euler2(const unsigned long n) {
	auto sum = 0.0f; // 1/1 + 1/4 + 1/9 + 1/16
	for (auto i = n; i > 0; i--)
		sum += 1.0 / (i * i);
	return sum;
}

int main() {
	for (auto n = 1UL; n <= 1'000'000; n *= 10) {
		cout << setw(12) << n << "\t" << setprecision(8) << sqrt(6 * euler1(n)) << "\t" << setprecision(8)
			 << sqrt(6 * euler2(n)) << endl;
	}
	return 0;
}
