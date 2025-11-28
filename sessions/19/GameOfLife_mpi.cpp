#include <cmath>
#include <cstdint>
#include <iostream>
#include <mpi.h>

using namespace std;

constexpr auto board_size = 10;
int world_size;
int world_rank;

int col, row; // your board location in the global world
// precomputed locations of the 8 neighboring worlds
int neighbor_east, neighbor_west, neighbor_north, neighbor_south;
int neighbor_north_east, neighbor_north_west;
int neighbor_south_east, neighbor_south_west;
class GameOfLife {
	uint8_t* board;
	uint8_t* next;
	uint8_t* leftbuffer;
	uint8_t* rightbuffer;
	int width;
	int height;
	int width2, height2; // handle edge cases +2
	int size;

public:
	GameOfLife(const int width, const int height) :
		width(width), height(height), width2(width + 2), height2(height + 2), size(width2 * height2) {
		board = new uint8_t[size];
		next = new uint8_t[size];
		leftbuffer = new uint8_t[size];
		rightbuffer = new uint8_t[size];
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
	int get(const int x, const int y) const { return board[y * width2 + x]; }

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
		for (auto i = 0, c = 0; i < height2; i++) {
			for (auto j = 1; j < width2; j++, c++)
				cout << get(i, j) << " ";
			c += 2;
			cout << endl;
		}
		cout << "===========================\n";
	}
	/*
	board               next

	BOARD 1
	0 0 0 0 0 0         0 0 0 0 0 0
	0 0 1 0 0 0         0 0 0 0 0 0
	0 0 1 0 0 0         0 1 1 1 0 0
	0 0 1 0 0 0         0 0 0 0 0 0
	0 0 y 0 0 0         0 0 0 0 0 0
	0 0 x 0 0 0         0 0 0 0 0 0

	0 0 y 0 0 0         0 0 0 0 0 0
	0 0 x 0 0 0         0 0 0 0 0 0
	0 0 1 0 0 0         0 1 1 1 0 0
	0 0 1 0 0 0         0 0 0 0 0 0
	0 0 0 0 0 0         0 0 0 0 0 0
	0 0 0 0 0 0         0 0 0 0 0 0
	BOARD 0
	*/

	void send_col(const int col, uint8_t* to, int to_neighbor) {
		for (int i = 0, ind = col + width2; i < height; i++, ind += width2)
			to[i] = board[ind];
		MPI_Isend(to, height, MPI_UNSIGNED_CHAR, to_neighbor, 0, MPI_COMM_WORLD);
	}
	void recv_col(const int col, uint8_t* from, int from_neighbor) {
		MPI_Recv(from, height, MPI_UNSIGNED_CHAR, from_neighbor, 0, MPI_COMM_WORLD);
		for (int i = 0, ind = col + width2; i < height; i++, ind += width2)
			board[ind] = from[i];
	}

	void step() {
		const int NORTH = -width2;
		constexpr int EAST = +1;
		const int SOUTH = +width2;
		constexpr int WEST = -1;
		const int NORTHEAST = NORTH + EAST;
		const int NORTHWEST = NORTH + WEST;
		const int SOUTHEAST = SOUTH + EAST;
		const int SOUTHWEST = SOUTH + WEST;
		// #define NOMPI
#if NOMPI
		MPI_ISend(board + width2 + 1, width, MPI_UNSIGNED_CHAR, neighbor_north, 0, MPI_COMM_WORLD);
		MPI_ISend(board + board_size * width2 + 1, width, MPI_UNSIGNED_CHAR, neighbor_south, 0, MPI_COMM_WORLD);
		send_col(1, leftbuffer, neighbor_west);
		send_col(width, rightbuffer, neighbor_east);
		// DIAGONALS! Go for it! homework

		MPI_recv(board + width2 * (height + 1) + 1, width, MPI_UNSIGNED_CHAR, neighbor_south, 0, MPI_COMM_WORLD);
		MPI_recv(board + 1, width, MPI_UNSIGNED_CHAR, neighbor_north, 0, MPI_COMM_WORLD);
		recv_col(0, leftbuffer, neighbor_west);
		recv_col(width2 - 1, rightbuffer, neighbor_east);


		if (world_rank == 0) {
			const int other = 1;
			// send the top edge of our board to the board "above us"
			MPI_Send(board + width2 + 1, width, MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD);
			// recv the top edge from the board "above us"
			MPI_Recv(board + 1, width, MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if (world_rank == 1) {
			const int other = 0;
			// receive the top edge of the board "above us"
			MPI_Recv(board + width2 * (height + 1) + 1, width, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD,
					 MPI_STATUS_IGNORE);
			MPI_Send(board + width2 * height + 1, width, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
		}
#endif
		for (int j = 0, c = width2 + 1; j < height; j++) {
			for (auto i = 0; i < width; i++, c++) {
				const int neighbors = board[c + EAST] + board[c + SOUTH] + board[c + WEST] + board[c + NORTH] +
					board[c + NORTHEAST] + board[c + NORTHWEST] + board[c + SOUTHEAST] + board[c + SOUTHWEST];
				if (board[c])
					next[c] = neighbors < 2 || neighbors > 3 ? 0 : 1;
				else
					next[c] = neighbors == 3 ? 1 : 0;
			}
			c += 2;
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

/*
topological map: torus
n=6
	0   1   2
	3   4   5

n = 16

	0  1  2  3
	4  5  6  7
	8  9  10 11
	12 13 14 15

*/


int main() {
	MPI_Init(nullptr, nullptr); // initialize MPI
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	const auto row_size = static_cast<int>(sqrt(world_size));
	const int max_col = row_size - 1;
	int row = world_rank / world_size;
	const int col = world_rank % world_size;
	// Codium is right except that the edges are not the same
	neighbor_east = col == max_col ? world_rank - max_col : world_rank + 1;
	neighbor_west = col == 0 ? world_rank + max_col : world_rank - 1;
	neighbor_north = row == 0 ? world_rank - row_size + world_size : world_rank - row_size;
	neighbor_south = row == max_row ? world_rank + row_size - world_size : world_rank + row_size;

	neighbor_north_east = col == max_col ? neighbor_north - max_col : neighbor_north + 1;
	neighbor_north_west = col == 0 ? neighbor_north + max_col : neighbor_north - 1;
	neighbor_south_east = col == max_col ? neighbor_south - max_col : neighbor_south + 1;
	neighbor_south_west = col == 0 ? neighbor_south + max_col : neighbor_south - 1;

	// initialize with 2 computers sharing NORTH/SOUTH border
	GameOfLife game(board_size, board_size); // n*n elements
	game.set2(1, 7); // 7*12 + 2
	game.print();
	for (auto i = 0; i < 10; i++) {
		game.step();
		game.print();
	}
	/*
	n=10 n^2 = 100 n = 10
	n = 1000 n^2 10^6 n = 1000
	n = 10,000 n^2 10^8
	*/
	MPI_Finalize();
	return 0;
}
