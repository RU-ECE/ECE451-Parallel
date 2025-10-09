#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

extern "C" {
    void read_1byte_sequentially(const char a[], uint64_t n);
    void read_1byte_sequentially_b(const char a[], uint64_t n);
    void read_8byte_sequentially(const uint64_t a[], uint64_t n);
    void read_8byte_skip(const uint64_t a[], uint64_t n, uint64_t skip);
}
/**
 * Read memory sequentially to measure memory performance.
 */
void read_memory(const char a[], uint64_t n) {
    volatile char sink;
    for (uint64_t i = 0; i < n; i++) {
        sink = a[i];
    }
}

/**
 * Read memory sequentially to measure memory performance 
 * 64 bits at a time.
 * THis will be faster, but not as much as you might think
 * Each read will take 1/8 the operations
 */
void read_memory64(const uint64_t a[], uint64_t n) {
    volatile uint64_t sink;
    for (uint64_t i = 0; i < n; i++) {
        sink = a[i];
    }
}

/**
 * Read memory with a given skip to measure memory performance.
 * This will be slower as the skip increases.
 */
void read_memory_skip(const uint64_t a[], uint64_t n, uint64_t skip) {
    volatile uint64_t sink;
    for (uint64_t j = 0; j < skip; j++) {
        for (uint64_t i = j; i < n; i+= skip) {
            sink = a[i];
        }
    }
}

int main() {
    const uint64_t n = 200'000'000;
    const uint64_t n8 = n*8;
    const uint64_t num_trials = 5;
    char *a = new char[n8]; // what is in here???
    auto t0 = high_resolution_clock::now();
    read_memory(a, n8);
    auto t1 = high_resolution_clock::now();
    cout << "reading bytes cold: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
    for (uint32_t trials = 0; trials < num_trials; trials++) {
        t0 = high_resolution_clock::now();
        read_memory(a, n8);
        t1 = high_resolution_clock::now();
        cout << "reading bytes warm: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
    }
    for (uint32_t trials = 0; trials < num_trials; trials++) {
        t0 = high_resolution_clock::now();
        read_1byte_sequentially(a, n8);
        t1 = high_resolution_clock::now();
        cout << "asm reading bytes: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
    }
    for (uint32_t trials = 0; trials < num_trials; trials++) {
        t0 = high_resolution_clock::now();
        read_1byte_sequentially_b(a, n8);
        t1 = high_resolution_clock::now();
        cout << "asm reading bytes b: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
    }
    delete[] a;

    void* p = malloc(n8*8);
    uint64_t *b = new uint64_t[n];
    t0 = high_resolution_clock::now();
    read_memory64(b, n);
    t1 = high_resolution_clock::now();
    cout << "reading 64-bit words cold: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
    for (uint32_t trials = 0; trials < num_trials; trials++) {
        t0 = high_resolution_clock::now();
        read_memory64(b, n);
        t1 = high_resolution_clock::now();
        cout << "reading 64-bit words warm: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
    }

    for (uint32_t trials = 0; trials < num_trials; trials++) {
        t0 = high_resolution_clock::now();
        read_8byte_sequentially(b, n);
        t1 = high_resolution_clock::now();
        cout << "asm 64-bit words: " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
    }
    for (uint64_t i = 2; i <= 1024; i*=2) {
        t0 = high_resolution_clock::now();
        read_memory_skip(b, n, i);
        t1 = high_resolution_clock::now();
        cout << "reading memory skip " << i << ": " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
    }
    for (uint64_t i = 2; i <= 1024; i*=2) {
        t0 = high_resolution_clock::now();
        read_8byte_skip(b, n/i, i);
        t1 = high_resolution_clock::now();
        cout << "reading memory skip " << i << ": " << duration_cast<microseconds>(t1 - t0).count() << " microseconds" << endl;
    }

    delete[] b;

    return 0;
}