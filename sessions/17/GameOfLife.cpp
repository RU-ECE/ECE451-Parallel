#include <iostream>
#include <mpi.h>

using namespace std;

class GameOfLife {
	unsigned char* board;
	unsigned char* next;
	int width;
	int height;
	int width2, height2; // handle edge cases +2
	int size;

public:
	GameOfLife(const int width, const int height) :
		width(width), height(height), width2(width + 2), height2(height + 2), size(width2 * height2) {
		board = new unsigned char[size];
		next = new unsigned char[size];
		for (auto i = 0; i < size; i++) {
			board[i] = 0;
			next[i] = 0;
		}
	}

	~GameOfLife() {
		delete[] board;
		delete[] next;
	}

	GameOfLife(const GameOfLife& other) = delete;
	GameOfLife& operator=(const GameOfLife& other) = delete;

	/*
	 * 0 0 0 0 0 0
	 * 0 x x x x 0
	 * 0 x x x x 0
	 * 0 x x x x 0
	 * 0 x x x x 0
	 * 0 0 0 0 0 0
	 *
	 * board				next
	 * 0 0 0 0 0 0			0 0 0 0 0 0
	 * 0 0 1 0 0 0			0 0 0 0 0 0
	 * 0 0 1 0 0 0			0 1 1 1 0 0
	 * 0 0 1 0 0 0			0 0 0 0 0 0
	 * 0 0 0 0 0 0			0 0 0 0 0 0
	 * 0 0 0 0 0 0			0 0 0 0 0 0
	 */
	void print() const {
		for (auto i = 1, c = width2 + 1; i < height2 - 1; i++) {
			for (auto j = 1; j < width2 - 1; j++, c++)
				cout << static_cast<int>(board[i * width + j]) << " ";
			c += 2;
			cout << endl;
		}
		cout << "===========================\n";
	}

	void step() {
		const auto NORTH = -width;
		constexpr auto EAST = +1;
		const auto SOUTH = +width;
		constexpr auto WEST = -1;
		const auto NORTHEAST = NORTH + EAST;
		const auto NORTHWEST = NORTH + WEST;
		const auto SOUTHEAST = SOUTH + EAST;
		const auto SOUTHWEST = SOUTH + WEST;
		for (int j = 0, c = width2 + 1; j < height; j++, c += 2) {
			for (auto i = 0; i < width; i++, c++) {
				const int neighbors = board[c + EAST] + board[c + SOUTH] + board[c + WEST] + board[c + NORTH] +
					board[c + NORTHEAST] + board[c + NORTHWEST] + board[c + SOUTHEAST] + board[c + SOUTHWEST];
				next[c] = board[c] ? neighbors < 2 || neighbors > 3 ? 0 : 1 : neighbors == 3 ? 1 : 0;
			}
		}
		swap(board, next); // just swap the pointers
	}

	void set(const int x, const int y) const { board[y * width + x] = 1; }
};

int main() {
	GameOfLife game(10, 10);
	game.set(7, 5);
	game.set(7, 6);
	game.set(8, 5);
	game.set(8, 6);
	game.set(2, 2);
	game.set(2, 3);
	game.set(2, 4);
	game.print();
	game.step();
	game.print();
	game.step();
	game.print();
	return 0;
}
