#include <thread>
#include <vector>

using namespace std;

vector<int> shared_list;
void add_to_list(int v) {
  for (int i = 0; i < 1'000'000; i++)
    shared_list.push_back(v);
}

int main(int argc, char *argv[]) {
    vector<int> a;
    a.push_back(1);
    a.push_back(19);
    a.push_back(3);

    thread t1(add_to_list, 5);
    thread t2(add_to_list, 6);
    t1.join();
    t2.join();
    return 0;
}
    