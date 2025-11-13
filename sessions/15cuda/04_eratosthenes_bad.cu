typedef uint32_t u32;

int prime_count = 0;

__global__
void eratosthenes(u32* primes, u32 n) {
    int tid = threadIdx.x; // each thread handles one element
    u32 count = 0; // where is this?? Who owns this?

// 65, 67, 69, 71, 73, 75, 77, 79, 81, 83
// t0 = 1..63, t1 = 65..127, ...

// 3,   3*3,  3*5, 3*7, ...
// 5,   5*5,  5*7, 5*9, ...


// here's a potentially better idea:
// let's all cooperate to write to memory[2, 1024]
// next thread can write [1025...2048]

// each thread will check numbers tid, tid+1, tid+2, ...
    u32 start = tid*64+1, end = start + 64;
    for (u32 i = start; i < end; i+=2) {
        if (primes[i]) { // divergent
            count++;
            for (u32 j = i * i; j <= n; j += i) {
                primes[j] = 0;
            }
        } else {
            // all threads NOT PRIME will sleep
        }
    }
}


__global__
void better_eratosthenes(u32* primes, u32 start, u32 end, u32 n) {
    int tid = threadIdx.x; // each thread handles one element
    u32 count = 0; // where is this?? Who owns this?
    __shared__ localstorage[256];
    u32 local[64]; // allocates this in registers?
}

int main() {
    const int n = 1'000'000;
    int *primes = new int[(n+31)/32];
    for (int i = 0; i < n; i++) {
        primes[i] = 1;
    }
    eratosthenes<<<(n+31)/32, 32>>>(primes, n);
    delete [] primes;

}