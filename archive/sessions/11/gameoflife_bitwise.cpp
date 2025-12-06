#include <iostream>

using namespace std;

/*
 *  00000000000000000000000000000000000000000000000000
 *  00000000000000000000000000000000000000000000000000
 *  00000000000000000000000000000000000000000000000000
 *
 *  case 1: if I, my neighbors to north, and south are all zero,
 *  then next time I am still all zero
 *
 *  00000000000000000000000000000000000000000000000000
 *  00000000000000000000000000000000000000000000000111
 *  00000000000000000000000000000000000000000000000000
 *
 *  spread the bits
 *      0  0  0  0  0  0  0
 *      0000000000000000000
 *      0  0  0  0  1  1  1
 *      0000000000000001000
 *
 *      1111         111
 *  1yx0         111
 *  1000         111
 *
 *      abc
 *       x
 *  NORTH, SOUTH, ME
 *
 *  Example bit-pos counting (pseudo):
 *		int bitpos(int i, uint64_t N, uint64_t ME, uint64_t S) {
 *  		int count = 0;
 *  		if (N & (1ULL << i))         count++;
 *  		if (N & (1ULL << (i-1)))     count++;
 *  		if (N & (1ULL << (i+1)))     count++;
 *  		if (S & (1ULL << i))         count++;
 *  		if (S & (1ULL << (i-1)))     count++;
 *  		if (S & (1ULL << (i+1)))     count++;
 *  		if (ME & (1ULL << (i+1)))    count++;
 *  		if (ME & (1ULL << (i-1)))    count++;
 *  		return count;
 *  	}
 *
 *  Masks and popcount idea:
 *  	mask = 0b111;
 *  	count = popcount(N & mask) + popcount(ME & mask) + popcount(S & mask);
 *
 *  In Verilog-like pseudocode:
 *  	// inputs: a_in[63:0], b_in[63:0], c_in[63:0]
 *  	for (int i = 0; i < 64; i++) {
 *  		count[i] = a_in[i] + a_in[i-1] + a_in[i+1]
 *  				 + b_in[i] + b_in[i-1] + b_in[i+1]
 *  				 + c_in[i] + c_in[i-1] + c_in[i+1];
 *  		// apply Game of Life rules per bit (alive vs dead)
 *  	}
 *
 */

unsigned int n;
unsigned long* life;
unsigned long* nextlife;

void setLife(const int i) { life[i / 64] |= 1ULL << (i % 64); }
bool isAlive(const int i) { return life[i / 64] &= 1ULL << (i % 64); }
bool isAlive(const unsigned int i) { return life[i / 64] &= 1ULL << (i % 64); }

void init() {
	const unsigned int size = (n + 2) * (n + 2);
	const unsigned int word_size = (size + 63) / 64;
	life = new unsigned long[word_size];
	nextlife = new unsigned long[word_size];

	for (auto i = 0U; i < word_size; i++)
		life[i] = 0;
	setLife(100);
	setLife(101);
	setLife(102);
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

void calcLiveOrDead(const unsigned int i) {
	constexpr auto EAST = +1U;
	constexpr auto WEST = -1U;
	const auto NORTH = -n - 2U;
	const auto SOUTH = +n + 2U;
	const auto count = isAlive(i + EAST) + isAlive(i + WEST) + // TODO: life[i+NORTH] + life[i+SOUTH] +
		life[i + NORTH + EAST] + life[i + NORTH + WEST] + life[i + SOUTH + EAST] + life[i + SOUTH + WEST];
	nextlife[i] = (isAlive(i) ? count == 2 || count == 3 : count == 3) ? 1 : 0;
}

void stepForward() {
	for (auto i = 0U, c = n + 2 + 1; i < n; i++, c += 2)
		for (auto j = 0U; j < n; j++, c++)
			calcLiveOrDead(c);
	swap(life, nextlife);
}

void print() {
	for (auto i = 0U, c = n + 2 + 1; i < n; i++, c += 2) {
		for (auto j = 0U; j < n; j++, c++)
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
