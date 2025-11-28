// imagine: 1 2 3 4 5 6 7 8  9 10 11 12 13 14 15 16
//          T1               T2

// instead:

// imagine: 1 2 3 4 5 6 7 8  9 10 11 12 13 14 15 16  17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
//          T1                                       T2


void multiply_arrays(const int* a, const int* b, int* c, const int size) {
	int temp = a[0] * b[0] - b[0] * a[0];

#pragma omp parallel for private(temp) schedule(static, 32)
	for (auto i = 0; i < size; ++i)
		c[i] = a[i] * b[i] - temp;
}
