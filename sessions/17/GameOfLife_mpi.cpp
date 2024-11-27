#include <mpi.h>
#include <cstdint>
#include <iostream>
using namespace std;

int world_size;
int world_rank;

class GameOfLife {
private:
    uint8_t* board;
    uint8_t* next;
    int width;
    int height;
    int width2, height2; // handle edge cases +2
    int size;
public:
    GameOfLife(int width, int height) 
    : width(width), height(height), 
    width2(width + 2), height2(height + 2), size(width2 * height2) {
        board = new uint8_t[size];
        next = new uint8_t[size];
        for (int i = 0; i < size; i++) {
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
	  int get(int x, int y) const {
		  return board[y * width2 + x];
	  }

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
        for (int i = 0, c = 0; i < height2; i++) {
            for (int j = 1; j < width2; j++, c++) {
							cout << get(i,j) << " ";
            }
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


    void step() {
        const int NORTH = -width2;
        const int EAST = +1;
        const int SOUTH = +width2;
        const int WEST = -1;
        const int NORTHEAST = NORTH + EAST;
        const int NORTHWEST = NORTH + WEST;
        const int SOUTHEAST = SOUTH + EAST;
        const int SOUTHWEST = SOUTH + WEST;
				//#define NOMPI
#if NOMPI
    if (world_rank == 0) {
        const int other = 1;
        // send the top edge of our board to the board "above us"
        MPI_Send(board+width2+1, width, MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD);
        // recv the top edge from the board "above us"
        MPI_Recv(board+1, width, MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank == 1) {
        const int other = 0;
        // receive the top edge of the board "above us"
        MPI_Recv(board+width2*(height+1)+1, width, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(board+width2*height+1, width, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }
#endif
        for (int j = 0, c = width2+1; j < height; j++) {
            for (int i = 0; i < width; i++, c++) {
                int neighbors =
                    board[c+EAST] + 
                    board[c+SOUTH] +
                    board[c+WEST] +
                    board[c+NORTH] +
                    board[c+NORTHEAST] +
                    board[c+NORTHWEST] +
                    board[c+SOUTHEAST] +
                    board[c+SOUTHWEST];
                if (board[c]) {
                    next[c] = neighbors < 2 || neighbors > 3 ? 0 : 1;
                } else {
                    next[c] = neighbors == 3 ? 1 : 0;
                }
            }
            c += 2;
        }
        swap(board, next); // just swap the pointers
    }
    void set(int x, int y) {
        board[y * width2 + x] = 1;
    }

    void set1();
    void set2(int x, int y);
};


void GameOfLife::set1() {
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
void GameOfLife::set2(int x, int y) {
    set(x+1,y);
    set(x+2,y+1);
    set(x,y+2);
    set(x+1,y+2);
    set(x+2,y+2);
}
int main() {
    MPI_Init(NULL, NULL); // initialize MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // initialize with 2 computers sharing NORTH/SOUTH border
    int n = 10;
    GameOfLife game(n, n); // n*n elements
    game.set2(1,7); // 7*12 + 2
    game.print();
    for (int i = 0; i < 10; i++) {
        game.step();
        game.print();
    }
/*
n=10   n^2 = 100   n = 10
n = 1000 n^2 10^6  n = 1000
n = 10,000 n^2 10^8
*/
    MPI_Finalize();
    return 0;
}
