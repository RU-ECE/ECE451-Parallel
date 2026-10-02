
extern "C" void sortoptimal14(int a[]);
void sort14(int a[], int n) {
    for (int i = 0; i < n; i += 14*8)
    sortoptimal14(a+i);
}

int main() {
    int a[128];
    for (int i = 0; i < 128; i++)
      a[i] = 128-i;
    sortoptimal14(a);
    return 0;
}