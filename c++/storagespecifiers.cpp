
int x; // x is global

void f() {
    int x = 2; // x is on the stack (auto)
    static int y = 2; // y is in global space, but not visible to other functions
    x++;
    y++;
    int* p = new int[100];
    delete[] p;
}

int main() {
    const double pi = 3.14159265358979323846;
    f();
    f();
    int x; // x is on the stack (auto)
    static int y; // y is in global space, but not visible to other functions
    int* p = new int[100];
    delete[] p;
}