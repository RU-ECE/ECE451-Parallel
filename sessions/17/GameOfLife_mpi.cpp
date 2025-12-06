#include <iostream>
#include <mpi.h>

using namespace std;

int world_size, world_rank;

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
	[[nodiscard]] int get(const int x, const int y) const { return board[y * width2 + x]; }

	/*
	0 0 0 0 0 0
	0 x x x x 0
	0 x x x x 0
	0 x x x x 0
	0 x x x x 0
	0 0 0 0 0 0

	board               next
	0 0 0 0 0 0         0 0 0 0 0 0
	0 0 1 0 0 0         0 0 0 0 0 0
	0 0 1 0 0 0         0 1 1 1 0 0
	0 0 1 0 0 0         0 0 0 0 0 0
	0 0 0 0 0 0         0 0 0 0 0 0
	0 0 0 0 0 0         0 0 0 0 0 0

	Suggestion 1: use top and bottom edge, not left and right
	Suggestion 2: copy the elements to a sequential buffer
	Suggestion 3: Use multiple MPI_Send
	*/

	void print() const {
		cout << "rank " << world_rank << endl;
		for (auto i = 0, c = 0; i < height2; i++, c += 2) {
			for (auto j = 1; j < width2; j++, c++)
				cout << get(i, j) << " ";
			cout << endl;
		}
		cout << "===========================\n";
	}

	/*
	 * board				next
	 *
	 * BOARD 1
	 * 0 0 0 0 0 0			0 0 0 0 0 0
	 * 0 0 1 0 0 0			0 0 0 0 0 0
	 * 0 0 1 0 0 0			0 1 1 1 0 0
	 * 0 0 1 0 0 0			0 0 0 0 0 0
	 * 0 0 y 0 0 0			0 0 0 0 0 0
	 * 0 0 x 0 0 0			0 0 0 0 0 0
	 *
	 * 0 0 y 0 0 0			0 0 0 0 0 0
	 * 0 0 x 0 0 0			0 0 0 0 0 0
	 * 0 0 1 0 0 0			0 1 1 1 0 0
	 * 0 0 1 0 0 0			0 0 0 0 0 0
	 * 0 0 0 0 0 0			0 0 0 0 0 0
	 * 0 0 0 0 0 0			0 0 0 0 0 0
	 * BOARD 0
	 */

	void step() {
		const auto NORTH = -width2;
		constexpr auto EAST = +1;
		const auto SOUTH = +width2;
		constexpr auto WEST = -1;
		const auto NORTHEAST = NORTH + EAST;
		const auto NORTHWEST = NORTH + WEST;
		const auto SOUTHEAST = SOUTH + EAST;
		const auto SOUTHWEST = SOUTH + WEST;
// #define NOMPI
#if NOMPI
		if (world_rank == 0) {
			const auto other = 1;
			// send the top edge of our board to the board "above us"
			MPI_Send(board + width2 + 1, width, MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD);
			// recv the top edge from the board "above us"
			MPI_Recv(board + 1, width, MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if (world_rank == 1) {
			const auto other = 0;
			// receive the top edge of the board "above us"
			MPI_Recv(board + width2 * (height + 1) + 1, width, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD,
					 MPI_STATUS_IGNORE);
			MPI_Send(board + width2 * height + 1, width, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
		}
#endif
		for (int j = 0, c = width2 + 1; j < height; j++, c += 2) {
			for (auto i = 0; i < width; i++, c++) {
				const int neighbors = board[c + EAST] + board[c + SOUTH] + board[c + WEST] + board[c + NORTH] +
					board[c + NORTHEAST] + board[c + NORTHWEST] + board[c + SOUTHEAST] + board[c + SOUTHWEST];
				next[c] = board[c] ? neighbors < 2 || neighbors > 3 ? 0 : 1 : neighbors == 3 ? 1 : 0;
			}
		}
		swap(board, next); // just swap the pointers
	}

	void set(const int x, const int y) const { board[y * width2 + x] = 1; }
	void set1() const;
	void set2(int x, int y) const;
};

void GameOfLife::set1() const {
	set(7, 5);
	set(7, 6);
	set(8, 5);
	set(8, 6);
	set(2, 2);
	set(2, 3);
	set(2, 4);
}

/*
  1,7
	x,y  *
			*
	 *   *  *

 */

// create a glider at x,y
void GameOfLife::set2(const int x, const int y) const {
	set(x + 1, y);
	set(x + 2, y + 1);
	set(x, y + 2);
	set(x + 1, y + 2);
	set(x + 2, y + 2);
}
int main() {
	MPI_Init(nullptr, nullptr); // initialize MPI
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	// initialize with 2 computers sharing NORTH/SOUTH border
	constexpr auto n = 10;
	GameOfLife game(n, n); // n*n elements
	game.set2(1, 7); // 7*12 + 2
	game.print();
	for (auto i = 0; i < 10; i++) {
		game.step();
		game.print();
	}
	/*
	 * n=10   n^2 = 100   n = 10
	 * n = 1000 n^2 10^6  n = 1000
	 * n = 10,000 n^2 10^8
	 */
	MPI_Finalize();
	return 0;
}
