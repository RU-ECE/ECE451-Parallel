#include <cmath>
#include <iostream>
#include <mpi.h>

using namespace std;

constexpr auto board_size = 10;
int world_size, world_rank;

int col, row; // your board location in the global world
// precomputed locations of the 8 neighboring worlds
int neighbor_east, neighbor_west, neighbor_north, neighbor_south, neighbor_north_east, neighbor_north_west,
	neighbor_south_east, neighbor_south_west;

class GameOfLife {
	unsigned char* board;
	unsigned char* next;
	unsigned char* leftbuffer;
	unsigned char* rightbuffer;
	int width;
	int height;
	int width2, height2; // handle edge cases +2
	int size;

public:
	GameOfLife(const int width, const int height) :
		width(width), height(height), width2(width + 2), height2(height + 2), size(width2 * height2) {
		board = new unsigned char[size];
		next = new unsigned char[size];
		leftbuffer = new unsigned char[size];
		rightbuffer = new unsigned char[size];
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

	void send_col(const int col, unsigned char* to, const int to_neighbor) const {
		for (auto i = 0, ind = col + width2; i < height; i++, ind += width2)
			to[i] = board[ind];
		MPI_Request req;
		MPI_Isend(to, height, MPI_UNSIGNED_CHAR, to_neighbor, 0, MPI_COMM_WORLD, &req);
		MPI_Wait(&req, MPI_STATUS_IGNORE);
	}
	void recv_col(const int col, unsigned char* from, const int from_neighbor) const {
		MPI_Recv(from, height, MPI_UNSIGNED_CHAR, from_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		for (auto i = 0, ind = col + width2; i < height; i++, ind += width2)
			board[ind] = from[i];
	}

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
				const auto neighbors = board[c + EAST] + board[c + SOUTH] + board[c + WEST] + board[c + NORTH] +
					board[c + NORTHEAST] + board[c + NORTHWEST] + board[c + SOUTHEAST] + board[c + SOUTHWEST];
				next[c] = board[c] ? neighbors < 2 || neighbors > 3 ? 0 : 1 : neighbors == 3 ? 1 : 0;
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
	const auto row_size = static_cast<int>(sqrt(world_size)), max_col = row_size - 1, max_row = row_size - 1,
			   row = world_rank / row_size, col = world_rank % row_size;
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
