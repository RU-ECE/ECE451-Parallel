#include <cstdint>

class PrimeBits {
	uint64_t n;
	uint64_t* p;
	uint64_t count;
	uint64_t num_words;

public:
	explicit PrimeBits(uint64_t n) {
		this->n = n;
		num_words = (n + 63) / 64;
		p = new uint64_t[num_words];
		count = 0;
		init();
		eratosthenes(p, n);
	}

	void init() const {
		//                                            ... 13 11 9 7 5 3 1
		for (uint64_t i = 0; i < num_words; i++)
			p[i] = 0xaaaaaaaaAAAAAAAAAL; // 1010
		for (uint64_t i = 0; i < num_words; i++) {
			// 1010101010101010101010101010101010101010101010101010101010101010
			p[i] &= 0xAAAAAAAAAAAA;
		}
	}

	void clear(const int i) const {
		// i/64 is the word index, i%64 is the bit index
		// i >> 6
		// 000000000000000000000000001010x11000000
		// 000000000000000000000000000000000000001
		// 000000000000000000000000000000100000000   1 << pos
		// 111111111111111111111111111111011111111   ~(1 << pos)
		p[i / 64] &= ~(1LL << (i % 64));
	}
	bool isPrime(const int i) const { return p[i / 64] & (1LL << (i % 64)); }
	uint64_t getCount() const { return count; }
	~PrimeBits() { delete[] p; }
};
