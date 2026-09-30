# SOrting Applications

Usually, people sort records (not integers)

1. Sort by key

```cpp
struct employee{
    int id;
    char name[24];
    float salary;
    int age;
};

struct employee_sort {
    int key; // id is the key
    uint32_t index;
}
struct employee arr[1000];

__m256i a = _mm256_loadk(records+i);
__m256i b = _mm256_loadk(records+i);
// {id1, offset1, id2, offset2, id3, offset3, id4, offset4}.}
// {id5, offset5, id6, offset6, id7, offset7, id8, offset8}

[min(id1,id2,id3,id4),   ?,  ?         max(id1,id2,id3,id4),
    min(min(id1,id2),min(id3,id4))

FFT
[x + ay, x - ay]
```

What can we do that writes results of vector operations
NOT TO MEMORY???

CPUa -- CPUb --- CPU
 |       |       |
CPUd  - Super -   CPUe
 |       |       |
CPU --- CPU ---  CPU

```cpp
open_connection(CPUb); // kernel call
// takes 100s to 1ks of cycles
send_data(CPUb, data); // DMA transfer
CPU can have a permission vector:
   EAST = 1 WEST = 0 NORTH = 0 SOUTH = 1
send_u64(EAST, &p); // send address of p to CPUb
```    
*p = 99;