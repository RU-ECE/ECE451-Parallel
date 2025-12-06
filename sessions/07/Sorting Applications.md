# Sorting Applications

Usually, we sort **records** (structs) rather than just integers.

---

## 1. Sort by Key

Example record type:

```c++
struct employee {
    int  id;
    char name[24];
    float salary;
    int  age;
};
````

To sort efficiently, we often separate the **key** from the full record and store an index (or pointer) to the record:

```c++
struct employee_sort {
    int      key;    // e.g., id is the key
    uint32_t index;  // index into the employee array
};

employee arr[1000];
```

We can then sort an array of `employee_sort` entries by `key`, and use `index` to refer back to `arr`.

---

## 2. Vectorized Sorting Sketch

Example idea using AVX vector instructions to operate on `(key, index)` pairs:

```c++
__m256i a = _mm256_loadk(records + i);
__m256i b = _mm256_loadk(records + i + 4);
// a might hold: {id1, offset1, id2, offset2, id3, offset3, id4, offset4}
// b might hold: {id5, offset5, id6, offset6, id7, offset7, id8, offset8}
```

Conceptually, we want to compute something like:

* `min(id1, id2, id3, id4)`
* `max(id1, id2, id3, id4)`
* And intermediates like `min(min(id1, id2), min(id3, id4))`, etc.

All done **within registers**, using vector compare and shuffle operations.

---

## 3. FFT Analogy

For a simple FFT butterfly on complex values:

* Inputs: $x$, $y$
* Output:

	* $x + a y$
	* $x - a y$

This is also a **vectorizable pattern** where both results can be computed in registers without immediately writing to
memory.

---

## 4. Avoiding Unnecessary Memory Writes

**Question:**
What can we do that writes results of vector operations **not directly to memory**, but keeps them in registers as long
as possible?

Ideas to think about:

* Reordering data using shuffle/blend instructions.
* Doing multiple stages of a sorting network while values remain in registers.
* Writing back to memory only when necessary (for example, after several compare–swap steps).

---

## 5. Network of CPUs (Conceptual)

Imagine a mesh of CPUs:

```text
CPUa -- CPUb --- CPU
  |       |      |
CPUd - Super - CPUe
  |       |      |
 CPU --- CPU --- CPU
```

We want a way for one CPU to **send data** directly to another.

Example pseudo-API:

```c++
open_connection(CPUb);       // kernel call
// May take hundreds to thousands of cycles

send_data(CPUb, data);       // DMA transfer to CPUb
```

A CPU can have a **permission vector**:

```text
EAST = 1
WEST = 0
NORTH = 0
SOUTH = 1
```

So we can do something like:

```c++
send_u64(EAST, &p);  // send the address of p to CPUb
```

Then, on the receiving CPU, we might do:

```c++
*p = 99;
```

This is a high-level sketch of **message passing / remote memory access** between CPUs using DMA and appropriate
permissions.
