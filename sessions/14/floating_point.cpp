#include <iomanip>
#include <iostream>

using namespace std;
/*
  1.23
	 .0887
   .769
	1
  1.23
	 .0887
========
  1
  1.31
   .769
  2.07
	11
	 .0887
   .769
=======
   .857
  1.23
  2.08

 */

int main() {
	auto sum = 0.0;
	for (auto x = 0; x < 100; x += 1) {
		cout << setprecision(8) << x / 10.0 << " ";
		sum += x / 10.0;
	}
	cout << "\nsum=" << setprecision(8) << sum << endl;
}
