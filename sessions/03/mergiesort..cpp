
/*
    1, 5, 6, 9                       3, 4, 11, 14
    i                                j

    1  3  4  5  6  9
       i                                j
                                           j
          i
                i
*/
void merge(const uint32_t a[], uint32_t[] b, uint32_t c[], uint32_t n) {
  int i = 0, j = 0, k = 0;
  while (i < n && j < n) {
    if (a[i] > b[j]) {
      c[k++] = b[j++];
    } else {
      c[k++] = a[i++];
    }
  }
}