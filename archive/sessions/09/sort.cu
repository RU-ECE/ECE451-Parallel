#include <cuda.h>
#include <iostream>

using namespace std;

inline void ordswap(int* a, const int i, const int j) {
	if (a[i] > a[j]) {
		const auto t = a[i];
		a[i] = a[j];
		a[j] = t;
	}
}

inline void ordswap2(int* a, const int i1, const int j1, const int i2, const int j2) {
	ordswap(a, i1, j1);
	ordswap(a, i2, j2);
}

inline void ordswap3(int* a, const int i1, const int j1, const int i2, const int j2, const int i3, const int j3) {
	ordswap2(a, i1, j1, i2, j2);
	ordswap(a, i3, j3);
}

inline void ordswap4(int* a, const int i1, const int j1, const int i2, const int j2, const int i3, const int j3,
					 const int i4, const int j4) {
	ordswap3(a, i1, j1, i2, j2, i3, j3);
	ordswap(a, i4, j4);
}

inline void ordswap8(int* a, const int i1, const int j1, const int i2, const int j2, const int i3, const int j3,
					 const int i4, const int j4, const int i5, const int j5, const int i6, const int j6, const int i7,
					 const int j7, const int i8, const int j8) {
	ordswap4(a, i1, j1, i2, j2, i3, j3, i4, j4);
	ordswap4(a, i5, j5, i6, j6, i7, j7, i8, j8);
}

/*
	[(0,2),(1,3),(4,6),(5,7)]
[(0,4),(1,5),(2,6),(3,7)]
[(0,1),(2,3),(4,5),(6,7)]
[(2,4),(3,5)]
[(1,4),(3,6)]
[(1,2),(3,4),(5,6)]
*/

inline void sort8network(int* a, const int i) {
	ordswap4(a, i, i + 2, i + 1, i + 3, i + 4, i + 6, i + 5, i + 7);
	ordswap4(a, i, i + 4, i + 1, i + 5, i + 2, i + 6, i + 3, i + 7);
	ordswap4(a, i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7);
	ordswap2(a, i + 2, i + 4, i + 3, i + 5);
	ordswap2(a, i + 1, i + 4, i + 3, i + 6);
	ordswap3(a, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6);
}

/*
optimal 16-way sorting network
*/
inline void sort16network(int* a, const int i) {
	/*
	[(0,13),(1,12),(2,15),(3,14),(4,8),(5,6),(7,11),(9,10)]
	[(0,5),(1,7),(2,9),(3,4),(6,13),(8,14),(10,15),(11,12)]
	[(0,1),(2,3),(4,5),(6,8),(7,9),(10,11),(12,13),(14,15)]
	[(0,2),(1,3),(4,10),(5,11),(6,7),(8,9),(12,14),(13,15)]
	*/
	ordswap8(a, i, i + 13, i + 1, i + 12, i + 2, i + 15, i + 3, i + 14, i + 4, i + 8, i + 5, i + 6, i + 7, i + 11,
			 i + 9, i + 10);
	ordswap8(a, i, i + 5, i + 1, i + 7, i + 2, i + 9, i + 3, i + 4, i + 6, i + 13, i + 8, i + 14, i + 10, i + 15,
			 i + 11, i + 12);
	ordswap8(a, i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 8, i + 7, i + 9, i + 10, i + 11, i + 12, i + 13,
			 i + 14, i + 15);
	ordswap8(a, i, i + 2, i + 1, i + 3, i + 4, i + 10, i + 5, i + 11, i + 6, i + 7, i + 8, i + 9, i + 12, i + 14,
			 i + 13, i + 15);
	//[(1,2),(3,12),(4,6),(5,7),(8,10),(9,11),(13,14)]
	ordswap4(a, i + 1, i + 2, i + 3, i + 12, i + 4, i + 6, i + 5, i + 7);
	ordswap3(a, i + 8, i + 10, i + 9, i + 11, i + 13, i + 14);
	//[(1,4),(2,6),(5,8),(7,10),(9,13),(11,14)]
	ordswap4(a, i + 1, i + 4, i + 2, i + 6, i + 5, i + 8, i + 7, i + 10);
	ordswap2(a, i + 9, i + 13, i + 11, i + 14);
	//[(2,4),(3,6),(9,12),(11,13)]
	ordswap4(a, i + 2, i + 4, i + 3, i + 6, i + 9, i + 12, i + 11, i + 13);
	//[(3,5),(6,8),(7,9),(10,12)]
	ordswap4(a, i + 3, i + 5, i + 6, i + 8, i + 7, i + 9, i + 10, i + 12);
	//[(3,4),(5,6),(7,8),(9,10),(11,12)]
	ordswap4(a, i + 3, i + 4, i + 5, i + 6, i + 7, i + 8, i + 9, i + 10);
	ordswap(a, i + 11, i + 12);
	//[(6,7),(8,9)]
	ordswap2(a, i + 6, i + 7, i + 8, i + 9);
}

void merge4(int dest[32], const int* a, const int* b, const int* c, const int* d) {
	auto i = 0, j = 0, k = 0, m = 0; // index of the 4 source arrays
	auto out = 0; // index of dest
	for (auto iter = 8; iter > 0; iter--) { // first 8 times, no one can be out of elements
		if (a[i] < a[j]) {
			if (a[k] < a[m])
				if (a[i] < a[k]) // a[i] is smallest case
					dest[out++] = a[i++];
				else // a[k] is smallest case
					dest[out++] = a[k++];
			else if (a[i] < a[m]) // a[i] is smallest case
				dest[out++] = a[i++];
			else // a[m] is smallest case
				dest[out++] = a[m++];
		} else {
			if (a[k] < a[m])
				if (a[j] < a[k]) // a[j] is smallest case
					dest[out++] = a[j++];
				else // a[k] is smallest case
					dest[out++] = a[k++];
			else if (a[j] < a[m]) // a[j] is smallest case
				dest[out++] = a[j++];
			else // a[m] is smallest case
				dest[out++] = a[m++];
		}
	}
	// now, one of the 4 arrays can be empty
	while (true) {
		if (a[i] < a[j]) {
			if (a[k] < a[m]) {
				if (a[i] < a[k]) { // a[i] is smallest case
					dest[out++] = a[i++];
					if (i >= 8) {
						a = d; // compact and use the last array as the first, keep going with only 3 arrays
						i = m;
						break;
					}
				} else { // a[k] is smallest case
					dest[out++] = a[k++];
					if (k >= 8) {
						k = m;
						c = d; // compact and use the last array as the 3rd, keep going with only 3 arrays
						break;
					}
				}
			} else if (a[i] < a[m]) { // a[i] is smallest case
				dest[out++] = a[i++];
				if (i >= 8) {
					i = m;
					a = d; // compact and use the last array as the first, keep going with only 3 arrays
					break;
				}
			} else { // a[m] is smallest case
				dest[out++] = a[m++];
				if (m >= 8)
					break;
			}
		} else {
			if (a[k] < a[m])
				if (a[j] < a[k]) // a[j] is smallest case
					dest[out++] = a[j++];
				else // a[k] is smallest case
					dest[out++] = a[k++];
			else if (a[j] < a[m]) // a[j] is smallest case
				dest[out++] = a[j++];
			else // a[m] is smallest case
				dest[out++] = a[m++];
		}
	}
	// at this point, only a,b,d remain, with i,j,k indexing
	while (true) {
		if (a[i] < a[j]) {
			if (a[i] < a[k]) { // a[i] is smallest case
				dest[out++] = a[i++];
				if (i >= 8) {
					i = k;
					a = c;
					break;
				}
			} else { // a[k] is smallest case
				dest[out++] = a[k++];
				if (k >= 8)
					break;
			}
		} else if (a[j] < a[k]) { // a[j] is smallest case
			dest[out++] = a[j++];
			if (j >= 8) {
				j = k;
				b = c;
				break;
			}
		} else { // a[k] is smallest case
			dest[out++] = a[k++];
			if (k >= 8)
				break;
		}
	}
	// onl6 a,b c remain, with i,j indexing the last 2 arrays
	while (true) {
		if (a[i] < a[j]) {
			dest[out++] = a[i++];
			if (i >= 8) {
				while (j < 8)
					dest[out++] = b[j++];
				break;
			}
		} else { // a[j] is smallest case
			dest[out++] = a[j++];
			if (j >= 8) {
				while (i < 8)
					dest[out++] = b[i++];
				break;
			}
		}
	}
}

/*
 Ordinary 2-way merge pass, which is taxing on global memory bandwidth
 question is, can we improve somehow by writing to local memory?
 "local" in CUDA is really "shared" global memory, and is not higher performance.
 registers can be used to store small local arrays of constant size, but this
 algorithm potentially writes to big blocks. Is it at all advantageous to write to
 a fixed-size buffer? Probably not.
*/
void merge2(int* dest, const int* a, const int* b, const int n) {
	auto i = 0, j = 0, k = 0;
	while (true) {
		if (a[i] < b[j]) {
			dest[k++] = a[i++];
			if (i >= n)
				while (j < n)
					dest[k++] = b[j++];
		} else {
			dest[k++] = b[j++];
			if (j >= n)
				while (i < n)
					dest[k++] = a[i++];
		}
	}
	for (auto i = 0; i < n; i++)
		dest[i + n] = b[i];
}

__global__ void sort(int* arr, const int n) {
	const auto i = n / threadIdx.x;
	// implement optimal 8-way sorting network on a-h
	// values all locally in registers
	int loc[32];
	// copy values of the array into local memory
	for (auto j = i; j < i + 8; j++)
		loc[j - i] = arr[j];
	sort8network(loc, 0);
	sort8network(loc, 8);
	sort8network(loc, 16);
	sort8network(loc, 24);
	// merge does not seem like it would work on SIMD processors, too many if statements
	// merge4(loc2, loc, loc + 8, loc + 16, loc + 24);
	for (auto j = 0; j < 32; j++)
		arr[i + j] = loc[j]; // copy back sorted groups into global memory
}

// initialize a to be backward
void init(int* a, const int n) {
	for (auto i = 0; i < n; i++)
		a[i] = n - i;
}

__global__ void gpu_init_random(int* a, const int n) {
	for (auto i = 0; i < n; i++)
		a[i] = rand() % n;
}

// initialize a to be random
void init_random(int* a, const int n) {
	for (auto i = 0; i < n; i++)
		a[i] = rand() % n;
}

int main() {
	constexpr auto n = 256U * 1024; // 1Gb (250M 4-byte integers)
	const auto arr = new int[n];
	init(arr, n);
	int* dev_arr;
	cudaMalloc(&dev_arr, n * sizeof(int));
	cudaMemcpy(dev_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
	// sort the array in groups of 8
	const auto t0 = clock();
	sort<<<1, 256>>>(dev_arr, n);
	cudaDeviceSynchronize();
	const auto t1 = clock();
	cout << "Elapsed time: " << (t1 - t0) << " ms" << endl;
	cudaMemcpy(arr, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
	for (auto i = 0U; i < 64; i += 8) {
		for (auto j = i; j < i + 8; j++)
			cout << arr[j] << '\t';
		cout << endl;
	}
	cudaFree(dev_arr);
	delete[] arr;
}
