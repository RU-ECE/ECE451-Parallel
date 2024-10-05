#include <iostream>
#include <unordered_map>

using namespace std;

class trie {
private:
  struct node {
    unordered_map<uint32_t,node*> next;
    uint32_t count;
    node() : count(0) {}
      bool contains(uint32_t a) {
          return (next.find(a) != next.end());      
     }
    void print(ostream& s) const {
      s << count << '\n';
      for (auto p : next) {
        p.second->print(s);
      }
    }
  };
  node root;
public:
  trie() {}
  void add(int a, int b) {
    node* p = &root;
    node* q = p->next[a];
    if (q == nullptr) {
      p->next[a] = new node();
      q = p;
      q->count++;
    }
    p->count++;
    p = p->next[b];
    p->count++;
  }
  void add(const uint32_t words[], uint32_t n) {
    node* p = &root;
    for (uint32_t i = 0; i < n; i++) {
      node* q = p->next[words[i]];
      if (q == nullptr) {
        p->next[words[i]] = q = new node();
        p = q;
      }
      p->count++;
    }
  }

  uint32_t contains(const uint32_t words[], uint32_t n) {
    node* p = &root;
    for (uint32_t i = 0; i < n; i++) {
      p = p->next[words[i]];
      if (p == nullptr) {
        return 0;
      }
    }
    return p->count;
  }

  friend ostream& operator <<(ostream& s, const trie& t) {
    const node* p = &t.root;
    p->print(s);
    return s;
  }
};

int main() {
    trie t;
    const uint32_t w1[] = {1,2,3};
    const uint32_t w2[] = {1,2,3,4};
    const uint32_t w3[] = {2, 3, 4, 5};
    const uint32_t w4[] = {2, 3, 5};
    t.add(w1, sizeof(w1)/sizeof(w1[0]));
    t.add(w2, sizeof(w2)/sizeof(w2[0]));
    t.add(w3, sizeof(w3)/sizeof(w3[0]));
    t.add(w4, sizeof(w4)/sizeof(w4[0]));
    cout << t << '\n';

    cout << t.contains(w1, sizeof(w1)/sizeof(w1[0])) << '\n';
    cout << t.contains(w1, sizeof(w1)/sizeof(w2[0])-1) << '\n';
    cout << t.contains(w2, sizeof(w2)/sizeof(w1[0])) << '\n';
    cout << t.contains(w3, sizeof(w3)/sizeof(w3[0])) << '\n';
    cout << t.contains(w4, sizeof(w4)/sizeof(w4[0])) << '\n';
    return 0;
}