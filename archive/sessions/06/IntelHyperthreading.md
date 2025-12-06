# Intel Hyperthreading

Assume a CPU with `n = 4` cores.

Each core has several **execution units**, for example:

- Integer add unit
- Integer multiply unit (for example, 3–4 clock cycles latency)
- Load/store unit (for memory operations)
- Instruction decode unit
- Vector AVX unit 1
- Vector AVX unit 2

---

## What Can Stall a CPU?

A core can stall for several reasons, including:

- **Waiting for memory**
	- Cache misses, high memory latency.
- **Waiting for instructions to load**
	- Instruction cache misses or decode bottlenecks.

When one thread is stalled, many of the execution units on the core may sit idle.

---

## What Does a Thread Need?

To run a thread, the CPU needs to maintain its architectural state:

- **Program Counter (PC)**
	- On x86-64: `rip` (instruction pointer), indicating where the code is.
- **Stack Pointer (SP)**
	- On x86-64: `rsp`, pointing to the thread’s stack.
- **Registers**
	- General-purpose registers (`rax`, `rbx`, `rcx`, $\ldots$), flags, etc.

---

## What Hyperthreading Adds

**Intel Hyper-Threading (SMT)** allows a single physical core to run **two hardware threads** by duplicating the
architectural state:

- Two sets of registers per core:
	- `rip` / `rip`
	- `rsp` / `rsp`
	- `rax` / `rax`
	- `rbx` / `rbx`
	- $\ldots$ and so on for the rest.

The physical execution units (ALUs, load/store units, vector units, etc.) are **shared**, but each logical thread has
its **own register set and PC**.

This enables:

- **Fast context switching** between the two hardware threads.
- If thread 1 is stalled (for example, waiting for memory), the core can quickly issue instructions from thread 2, so:
	- Execution units unused by thread 1 can be used by thread 2.

---

## Hashing Aside

A common pattern for hashing with a table of size `n` (where `n` is a power of 2):

```c++
index = f(x) % n;     // hash(x) = f(x) mod n
// If n is a power of 2, this can be done as:
index = f(x) & (n - 1);
````

Using `& (n - 1)` is often faster than `% n` when `n` is a power of two.