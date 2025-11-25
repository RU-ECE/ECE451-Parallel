//sorting

void sort8cols(__m256i a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h) {
    if (a > b) swap(a, b);
    if (b > c) swap(b, c);
    if (c > d) swap(c, d);
    if (d > e) swap(d, e);
    if (e > f) swap(e, f);
    if (f > g) swap(f, g);
    if (g > h) swap(g, h);
} 

void sort16cols(__m256i a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h) {
   
} 

void transpose8() {

}



void merge(a, b, c, d) {

}

void sort(int a[], int n) {

    if (a[i] > a[j])
      swap(a[i], a[j]);
}