#include <iostream>
#include <ranges>
#include <unordered_map>

using namespace std;

class trie {
	struct node {
		unordered_map<unsigned int, node*> next;
		unsigned int count;
		node() : count(0) {}
		bool contains(const unsigned int a) const { return next.contains(a); }
		void print(ostream& s) const {
			s << count << endl;
			for (const auto snd : next | views::values)
				snd->print(s);
		}
	};
	node root;

public:
	trie() = default;
	void add(const int a, const int b) {
		auto p = &root;
		if (auto q = p->next[a]; q == nullptr) {
			p->next[a] = new node();
			q = p;
			q->count++;
		}
		p->count++;
		p = p->next[b];
		p->count++;
	}
	void add(const unsigned int words[], const unsigned int n) {
		auto p = &root;
		for (auto i = 0U; i < n; i++) {
			if (auto q = p->next[words[i]]; q == nullptr) {
				p->next[words[i]] = q = new node();
				p = q;
			}
			p->count++;
		}
	}

	unsigned int contains(const unsigned int words[], const unsigned int n) {
		auto p = &root;
		for (auto i = 0U; i < n; i++) {
			p = p->next[words[i]];
			if (p == nullptr)
				return 0;
		}
		return p->count;
	}

	friend ostream& operator<<(ostream& s, const trie& t) {
		const auto p = &t.root;
		p->print(s);
		return s;
	}
};

int main() {
	trie t;
	constexpr unsigned int w1[] = {1, 2, 3};
	constexpr unsigned int w2[] = {1, 2, 3, 4};
	constexpr unsigned int w3[] = {2, 3, 4, 5};
	constexpr unsigned int w4[] = {2, 3, 5};
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
