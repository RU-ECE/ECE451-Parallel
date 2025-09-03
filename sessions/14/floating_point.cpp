#include <iostream>
#include <iomanip>
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
	float sum = 0;
  for (float x = 0; x < 10; x += 0.1) {
		cout << setprecision(8) << x << " ";
		sum += x;
	}
	cout << "\nsum="<<  setprecision(8)  << sum << '\n';

}
