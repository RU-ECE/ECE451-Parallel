// 1 2 4 5 6 7 8 10              9 10 11 12 13 14 15 16
//                  i                j
// out = 1 2 4 5 6 7 8 9         10 10 
void merge(uint32_t* a, uint32_t* b, uint32_t* out, uint32_t n) {
    int i = 0, j = 0, k = 0;
    while (i < n && j < n) {
        if (a[i] > b[j]) {
            out[k++] = b[j++];
        } else {
            out[k++] = a[i++];
        }
    }
    while (i < n) {
        out[k++] = a[i++];
    }
    while (j < n) {
        out[k++] = b[j++];
    }
}

// note: these arrays might not be aligned. It's faster if they are
void mergeavx(uint32_t* a, uint32_t* b, uint32_t* out, uint32_t n) {
    _m256i a0 = _m256i_loadu(a);
    _m256i a1 = _m256i_loadu(a + 8);
    _m256i a2 = _m256i_loadu(a + 16);
    _m256i a3 = _m256i_loadu(a + 24);

    _m256i b0 = _m256i_loadu(b);
    _m256i b1 = _m256i_loadu(b+8);
    _m256i b2 = _m256i_loadu(b+16);
    _m256i b3 = _m256i_loadu(b+24);

    vmin_u32(a0, b0)
    // pull out the smallest value, write into out0

    // instead, can we pull out the SMALLEST 2 values?
    // a0 = [a0v1 a0v2 a0v3 a0v4 a0v5 a0v6 a0v7 a0v8]
    // b0 = [b0v1 b0v2 b0v3 b0v4 b0v5 b0v6 b0v7 b0v8]

    // your job [ a0 b0 ...] or [b0 a0 ...] OR [b0 b1] or [a0 a1]
    _m256i out0;
    _m256i out1;
//    _m256i out2;
//    _m256i out3;


}
 
/*
 Merge sort is not perfectly parallel

 At first...
n = 10^6

merge every 8 to 16 1M/8 = 125K ==> 62.5k
merge every 16 to 32  1M/32 = 31.25k


...
merge every 1024 to 2048 = 1k

... 
merge every 32k to 64k

merge 2 lists of 250k each to 500k (4)
at the end merge 2 lists of 500k each into 1 (not parallel at all)

instead of doing merge 2
can we merge 4?

a = [ 2....]
b = [ 1...]
c = [ 5...]
d = [ 6...]

challenge: Trying to come up with vector architecture that can do this
generally, NOT JUST FOR SORTING


*/

