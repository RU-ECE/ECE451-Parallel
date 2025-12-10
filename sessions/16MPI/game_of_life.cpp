#include <mpi.h>

#include <vector>
#include <iostream>

/*
	Conway's Game of Life
	Designed in class by ECE451/566

	given: 2d grid of cells

     n n n
     n c n
     n n n

1. we must make sure code works at the edge of the world
     n n 0
     n c 0
     n n 0
  solution: create an edge all around so no cell is at the "edge"

2. If you change c while you are computing, the next cell over is wrong 
     n n n n
     n c n n
     n n n n

3. for MPI, we have to exchange data at each edge. The same edge trick works
       0          1
    c c c c    c c c c
    c c c c    c c c c
    c c c c    c c c c
    c c c c    c c c c
    c c c c    c c c c

    c c c c    c c c c
    c c c c    c c c c
    c c c c    c c c c
    c c c c    c c c c
    c c c c    c c c c
       2          3


       0                1                2
  e e e e e e      e e e e e e      e e e e e e
  e c c c c e      e c c c c e      e c c c c e
  e c c c c e      e c c c c e      e c c c c e
  e c c c c e      e c c c c e      e c c c c e
  e c c c c e      e c c c c e      e c c c c e
  e c c c c e      e c c c c e      e c c c c e
  e e e e e e      e e e e e e      e c c c c e

       3                4                5
  e e e e e e      e e e e e e      e e e e e e
  e c c c c e      e c c c c e      e c c c c e
  e c c c c e      e c c c c e      e c c c c e
  e c c c c e      e c c c c e      e c c c c e
  e c c c c e      e c c c c e      e c c c c e
  e c c c c e      e c c c c e      e c c c c e
  e e e e e e      e e e e e e      e c c c c e

       6                7                8
  e e e e e e      e e e e e e      e e e e e e
  e c c c c e      e c c c c e      e c c c c e
  e c c c c e      e c c c c e      e c c c c e
  e c c c c e      e c c c c e      e c c c c e
  e c c c c e      e c c c c e      e c c c c e
  e c c c c e      e c c c c e      e c c c c e
  e e e e e e      e e e e e e      e c c c c e



 */

uint8_t* gol_map;


class GameOfLife {
private:
  uint8_t* cells;
	uint8_t* next;

	// NOT INCLUDING EDGES/Ghosts
	uint32_t rows; // number of rows in the map of cells on this process
	uint32_t cols; // number of rows in the map of cells on this process
	uint32_t size;
	uint32_t rank; // your process number in the mess 

	uint32_t process_rows; // the number of processes in each row
	uint32_t process_cols;
public:
  GameOfLife(uint32_t rows, uint32_t cols,
						 uint32_t process_rows, uint32_t process_cols);
	~GameOfLife() {
		delete [] cells;
		delete [] next;
	}
	void exchange_neighbors();
	void step_forward();
	void print() const;
};

GameOfLife::GameOfLife(uint32_t rows, uint32_t cols,
											 uint32_t process_rows, uint32_t process_cols,
											 int rank,
											 const char filename[])
	: rows(rows), cols(cols), process_rows(process_rows), process_cols(process_cols), rank(rank) {
	size = (rows+2)*(cols+2);
  cells = new uint8_t[size];
	next = new uint8_t[size];

	if (rank == 0) {
		load(filename);
		golmap = new uint8_t[rows*process_rows*cols*process_cols];
  } else {
		for (int i = 0; i < size; i++)
			cells[i] = 0;
		for (int i = 0; i < size; i++)
			next[i] = 0;
  }
}


int main() {
    MPI_Init(nullptr, nullptr);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

		const int rows = 4, cols = 5, process_rows = 3, process_cols = 3;
		const int generations = 30;
		
		GameOfLife g(rows, cols, process_rows, process_cols, world_rank, "gol.txt");
		for (int t = 0; t < generations; t++) {
			g.exchange_neighbors();
			g.step_forward();
      g.print(golmap);
		}
		
		
    MPI_Finalize();
    return 0;
}
