#include <iostream>
#include <unistd.h>
#include <thread>
using namespace std;

int count_bananas = 0;

void grow_bananas(int n) {
  for (int i = 0; i < n; i++) {
    count_bananas++; // read count_bananas, +1, write back
  }
}

void eat_bananas(int n) {
  for (int i = 0; i < n; i++) {
    count_bananas--; // read count_bananas, -1, write back
   }
}

const int n = 10;

void threading1() {
  thread t1(grow_bananas, n);
  thread t2(eat_bananas, n);
  cout << count_bananas << endl;
}

void threading2() {
  thread t1(grow_bananas, n);
  t1.join();
  thread t2(eat_bananas, n);
  t2.join();
  cout << count_bananas << ' ';
}
void threading3() {
  thread t1(grow_bananas, n);
  thread t2(eat_bananas, n);
  t1.join();
  t2.join();
  cout << count_bananas << endl;
}

int main() {
  //threading1();
  //threading2();
  threading3();
}

