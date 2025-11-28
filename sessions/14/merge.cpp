// 8 7 6 5 4 3 2 1
// 7 8 5 6 3 4 1 2
// in = 5 6 7 8 1 2 3 4
//             i       j
// 1 2 3 4 5 6 7 8
// out = [1 2 3 4 5 6 7 8 ]

void merge(const int* in, int* out, const int n) {
	const int right = n;
	const int end = n + n;
	int i = 0, j = right;
	auto k = 0;
	while (i < n && j < end)
		out[k++] = in[(in[i] < in[j] ? i : j)++];
	while (i < n)
		out[k++] = in[i++];
	while (j < n)
		out[k++] = in[j++];
}
