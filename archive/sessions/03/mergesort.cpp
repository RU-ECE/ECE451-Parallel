/*
	1, 5, 6, 9                       3, 4, 11, 14
	i                                j

	1  3  4  5  6  9
	   i                                j
										   j
		  i
				i
*/

void merge(const unsigned int a[], const unsigned int b[], unsigned int c[], const unsigned int n) {
	auto i = 0U, j = 0U, k = 0U;
	while (i < n && j < n)
		c[k++] = a[i] > b[j] ? b[j++] : a[i++];
}

void merge4(const unsigned int a[], const unsigned int b[], const unsigned int c[], const unsigned int d[], unsigned int out[],
			unsigned int n) {}
