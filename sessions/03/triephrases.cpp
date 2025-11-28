#include <cstdint>
#include <iostream>
#include <ranges>
#include <unordered_map>

using namespace std;

class trie {
	struct node {
		unordered_map<uint32_t, node*> next;
		uint32_t count;
		node() : count(0) {}
		bool contains(const uint32_t a) const { return next.contains(a); }
		void print(ostream& s) const {
			s << count << endl;
			for (const auto snd : next | views::values)
				snd->print(s);
		}
	};
	node root;

public:
	trie() {}
	void add(const int a, const int b) {
		node* p = &root;
		if (node* q = p->next[a]; q == nullptr) {
			p->next[a] = new node();
			q = p;
			q->count++;
		}
		p->count++;
		p = p->next[b];
		p->count++;
	}
	void add(const uint32_t words[], const uint32_t n) {
		node* p = &root;
		for (uint32_t i = 0; i < n; i++) {
			if (node* q = p->next[words[i]]; q == nullptr) {
				p->next[words[i]] = q = new node();
				p = q;
			}
			p->count++;
		}
	}

	uint32_t contains(const uint32_t words[], const uint32_t n) {
		node* p = &root;
		for (uint32_t i = 0; i < n; i++) {
			p = p->next[words[i]];
			if (p == nullptr)
				return 0;
		}
		return p->count;
	}

	friend ostream& operator<<(ostream& s, const trie& t) {
		const node* p = &t.root;
		p->print(s);
		return s;
	}
};

int main() {
	trie t;
	constexpr uint32_t w1[] = {1, 2, 3};
	constexpr uint32_t w2[] = {1, 2, 3, 4};
	constexpr uint32_t w3[] = {2, 3, 4, 5};
	constexpr uint32_t w4[] = {2, 3, 5};
	t.add(w1, size(w1));
	t.add(w2, size(w2));
	t.add(w3, size(w3));
	t.add(w4, size(w4));
	cout << t << endl;

	cout << t.contains(w1, size(w1)) << endl;
	cout << t.contains(w1, size(w1) - 1) << endl;
	cout << t.contains(w2, size(w2)) << endl;
	cout << t.contains(w3, size(w3)) << endl;
	cout << t.contains(w4, size(w4)) << endl;
	return 0;
}
