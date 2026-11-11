all: ompsimd omp

ompsimd:
	g++ -O3 -g main.cpp -mavx2 -fopenmp ompmult.cpp -o ompsimd

omp:
	g++ -O3 -g main.cpp -o omp
