/*
    merge two sorted lists a and b of size n into one list c of size 2n


    1, 2, 5, 9
          i
    
    2, 4, 6, 8
          j

    c holds 8 elements
    [1, 2, 2, 4, 0, 0, 0, 0]
              k
*/

void merge(const int* a, const int* b, int* c, int n) {
    int i = 0
    for (int k = 0; k < 2*n; k++) {
        if (a[i] < b[j]) {
           c[k] = a[i];
           i++;
            if (i >= n) {
                while (j < n) {
                    c[k] = b[j];
                    j++;
                    k++;
                }
                break;
            }
        } else [
            c[k] = b[j];
            j++;
            if (j >= n) {
                while (i < n) {
                    c[k] = a[i];
                    i++;
                    k++;
                }
                break;
            }
        ]
    }
}