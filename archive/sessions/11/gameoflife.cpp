#include <cstdint>
#include <iostream>

using namespace std;

uint32_t n;
uint8_t* life;
uint8_t* nextlife;


void init() {
	const uint32_t size = (n + 2) * (n + 2);
	life = new uint8_t[size];
	nextlife = new uint8_t[size];

	for (auto i = 0; i < size; i++)
		life[i] = 0;
}

/*
	vector approach

	10010001


	   1    0    0    1    0    0    0    1
	0000 0000 0000 0000 0000 0000 0000 0000

	00000000000000000000000000000000
	00000000000000000000110000000000
	00000000000000000000110000000000
	00000000000000000000000000000000


*/

void calcLiveOrDead(const uint32_t i) {
	constexpr uint32_t EAST = +1;
	constexpr uint32_t WEST = -1;
	const uint32_t NORTH = -n - 2;
	const uint32_t SOUTH = +n + 2;
	const int count = life[i + EAST] + life[i + WEST] + life[i + NORTH] + life[i + SOUTH] + life[i + NORTH + EAST] +
		life[i + NORTH + WEST] + life[i + SOUTH + EAST] + life[i + SOUTH + WEST];
	nextlife[i] = life[i] ? count >= 2 && count <= 3 : count == 3;
}

void stepForward() {

	for (int i = 0, c = n + 2 + 1; i < n; i++, c += 2)
		for (auto j = 0; j < n; j++, c++)
			calcLiveOrDead(c);
	swap(life, nextlife);
}

void print() {
	for (int i = 0, c = n + 2 + 1; i < n; i++, c += 2) {
		for (auto j = 0; j < n; j++, c++)
			cout << static_cast<int>(life[c]) << ' ';
		cout << "\n";
	}
	cout << "\n\n";
}
int main() {
	n = 10;
	const int row = n + 2;
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
