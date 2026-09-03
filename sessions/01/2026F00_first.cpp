#include <iostream>

using namespace std;

uint64_t sum(uint32_t a, uint32_t b) {
	uint64_t s = 0;
	for (int i = a; i <= b; i++)
		s += i;
	return s;
}

int fact(int n) {
	if (n <= 0)
		return 1;
	return n * fact(n-1);
}

int main() {
	int yy[200] = {9};
	cout << fact(19);
	cout << "hello\n";
	int a, b;
	cout << "Enter two numbers: ";
	cin >> a >> b;
	cout << sum(a, b) << '\n';
}
