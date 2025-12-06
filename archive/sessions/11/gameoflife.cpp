#include <iostream>

using namespace std;

int n;
unsigned char* life;
unsigned char* nextlife;

void init() {
	const auto size = (n + 2U) * (n + 2);
	life = new unsigned char[size];
	nextlife = new unsigned char[size];
	for (auto i = 0U; i < size; i++)
		life[i] = 0;
}

/*
 * vector approach
 *
 * 10010001
 *
 *   1    0    0    1    0    0    0    1
 * 0000 0000 0000 0000 0000 0000 0000 0000
 *
 * 00000000000000000000000000000000
 * 00000000000000000000110000000000
 * 00000000000000000000110000000000
 * 00000000000000000000000000000000
 */

void calcLiveOrDead(const unsigned int i) {
	constexpr auto EAST = +1;
	constexpr auto WEST = -1;
	const auto NORTH = -n - 2;
	const auto SOUTH = +n + 2;
	const auto count = life[i + EAST] + life[i + WEST] + life[i + NORTH] + life[i + SOUTH] + life[i + NORTH + EAST] +
		life[i + NORTH + WEST] + life[i + SOUTH + EAST] + life[i + SOUTH + WEST];
	nextlife[i] = life[i] ? count >= 2 && count <= 3 : count == 3;
}

void stepForward() {
	for (auto i = 0, c = n + 2 + 1; i < n; i++, c += 2)
		for (auto j = 0; j < n; j++, c++)
			calcLiveOrDead(c);
	swap(life, nextlife);
}

void print() {
	for (auto i = 0, c = n + 2 + 1; i < n; i++, c += 2) {
		for (auto j = 0; j < n; j++, c++)
			cout << life[c] << ' ';
		cout << endl;
	}
	cout << "\n\n";
}
int main() {
	n = 10;
	const auto row = n + 2;
	constexpr auto num_generations = 4;
	init();
	life[2 * row + 3] = 1;
	life[2 * row + 4] = 1;
	life[2 * row + 5] = 1;
	for (auto i = 0; i < num_generations; i++) {
		stepForward();
		print();
	}
}
