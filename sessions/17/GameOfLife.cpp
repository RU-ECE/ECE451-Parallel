#include <mpi.h>
#include <cstdint>
#include <iostream>
using namespace std;

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

*/
    void print() const {
        for (int i = 1, c = width2+1; i < height2-1; i++) {
            for (int j = 1; j < width2-1; j++, c++) {
                cout << int(board[i * width + j]) << " ";
            }
            c += 2;
            cout << endl;
        }
        cout << "===========================\n";
    }
    void step() {
        const int NORTH = -width;
        const int EAST = +1;
        const int SOUTH = +width;
        const int WEST = -1;
        const int NORTHEAST = NORTH + EAST;
        const int NORTHWEST = NORTH + WEST;
        const int SOUTHEAST = SOUTH + EAST;
        const int SOUTHWEST = SOUTH + WEST;

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
        board[y * width + x] = 1;
    }
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