#include <iostream>


int x = 1; //global

int main() {
    int x = 2; // auto (on stack)
    ::x = 3; // global
    x = 5;
    if (2 < 3) {
        int x = 6;
    }
}