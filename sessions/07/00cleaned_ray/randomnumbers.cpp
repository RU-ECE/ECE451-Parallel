#include <cstdlib>
#include <random>
#include <cstdint>
#include <iostream>

using namespace std;

int main() {

	int a = rand(); // old garbage. DO NOT USE
	int b = rand(); // old garbage. DO NOT USE
	
  // rand uses a linear congruential rng	 "Numerical recipes"

	// a --> b
	// pairs of numbers from rand() are not very random. They form lines

	//	mt19937 rng(random_device{}());
	mt19937 rng(1);

	// Uniform integer: [1, 100], inclusive
	uniform_int_distribution<int> int_dist(1, 100);

	// Uniform double: [0.0, 1.0)
	uniform_real_distribution<double> double_dist(0.0, 1.0);

	// Normal distribution: mean = 100.0, standard deviation = 15.0
	normal_distribution<double> normal_dist(100.0, 15.0);

	cout << "Integer: " << int_dist(rng) << '\n';
	cout << "Double:  " << double_dist(rng) << '\n';
	cout << "Normal:  " << normal_dist(rng) << '\n';

    return 0;
}
