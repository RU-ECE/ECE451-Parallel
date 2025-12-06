class PrimeBits {
	unsigned long n;
	unsigned long* p;
	unsigned long count;
	unsigned long num_words;

public:
	explicit PrimeBits(unsigned long n) {
		this->n = n;
		num_words = (n + 63) / 64;
		p = new unsigned long[num_words];
		count = 0;
		init();
		eratosthenes(p, n);
	}

	void init() const {
		// ... 13 11 9 7 5 3 1
		for (auto i = 0UL; i < num_words; i++)
			p[i] = 0xAUL; // 1010
		for (auto i = 0UL; i < num_words; i++) {
			// 1010101010101010101010101010101010101010101010101010101010101010
			p[i] &= 0xAAAAAAAAAAAAAAAAL;
		}
	}

	void clear(const int i) const {
		// i/64 is the word index, i%64 is the bit index
		// i >> 6
		// 000000000000000000000000001010x11000000
		// 000000000000000000000000000000000000001
		// 000000000000000000000000000000100000000 1 << pos
		// 111111111111111111111111111111011111111 ~(1 << pos)
		p[i / 64] &= ~(1LL << (i % 64));
	}
	[[nodiscard]] bool isPrime(const int i) const { return p[i / 64] & 1LL << (i % 64); }
	[[nodiscard]] unsigned long getCount() const { return count; }
	~PrimeBits() { delete[] p; }
};
