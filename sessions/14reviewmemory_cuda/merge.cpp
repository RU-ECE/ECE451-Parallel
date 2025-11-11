// 8 7 6 5 4 3 2 1
// 7 8 5 6 3 4 1 2
// in = 5 6 7 8 1 2 3 4
//             i       j
// 1 2 3 4 5 6 7 8
// out = [1 2 3 4 5 6 7 8 ]

void merge(int* in, int*out, int n) {
	int right = n;
	int end = n+n;
	int i = 0, j = right;
	int k = 0;
  while (i < n && j < end) {
		if (in[i] < in[j]) {
			out[k++] = in[i++];
		} else {
			out[k++] = in[j++];
		}
	}
	while (i < n)
		out[k++] = in[i++];
	while (j < n)
		out[k++] = in[j++];
}
