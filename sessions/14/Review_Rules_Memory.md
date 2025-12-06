# Rules of Memory Access

Assume:

- 2 banks of DDR5 memory (burst length 16).
- Cache line size = 64 bytes.
- Each element is 8 bytes (e.g., `uint64_t`).
- So reading one cache line brings in 8 elements.

When reading from RAM we always bring in a **cache line**:

- If you read memory location `x` such that $x \bmod 64 = 0$:
	- You actually read: `x, x+8, x+16, x+24, ...` – 8 elements = 64 bytes (one cache line).
	- With 2 banks and burst 16, you effectively get 32 elements per CAS burst in the simplified model.

---

## 1. Random vs Row Hit

Assume timing (from DDR5 example):

- RAS = 46 clocks
- CAS = 45 clocks
- Burst = 16 (we model this as +15 extra cycles after CAS)

1. **To read an arbitrary location (new row):**

   $$\text{RAS} + \text{CAS} + 15 = 46 + 45 + 15 = 106 \text{ clocks}$$

2. **To read within the current row:**

   $$\text{CAS} + 15 = 45 + 15 = 60 \text{ clocks}$$

Example loop reading from L1 cache (conceptual):

```asm
1:
    mov    (%r8), %rax      # READ FROM CACHE L1
    sub    $1, %rdi
    jg     1b
````

* L1 cache latency ≈ 4 clocks.

---

## 2. Sequential Access Going Backwards

Read from location `a[rdi]`, moving backward:

```asm
1:
    mov    (%r8, %rdi, 8), %rax   # READ FROM CACHE L1 (backwards)
    sub    $1, %rdi
    jg     1b
```

* CAS + 15 = 60 clocks to read 32 elements (2 banks × 16 burst each).

* Effective cost:

  $$
  \frac{60}{32} \approx 1.875 \text{ clocks per element}
  $$

* Random new-row access:

  $$\text{RAS} + \text{CAS} + 16 \approx 46 + 45 + 16 = 107 \text{ clocks}$$

(Using 16 rather than 15 in this line is just a slightly different model.)

---

## 3. Sequential Access Forwards

```asm
1:
    mov    (%r8), %rax      # READ FROM CACHE L1 (forwards)
    add    $8, %r8          # move to next element
    sub    $1, %rdi
    jg     1b
```

Key idea:

> You can **never** use cache if you did not read from it before.

But once the first access brings data into cache (and prefetch kicks in), sequential forward access can be very
efficient.

---

## 4. Strided Access – Skipping Elements

### 4.1. Skip 2 Elements (stride = 16 bytes)

```asm
1:
    mov    (%r8), %rax      # READ FROM CACHE L1
    add    $16, %r8         # skip 2 elements (each 8 bytes)
    sub    $1, %rdi
    jg     1b
```

* CAS + 15 = 60 clocks read 32 elements, but we only **use** every 2nd element.
* Effective cost:

  $$
  \frac{60}{16} = 3.75 \text{ clocks per element (used)}
  $$

### 4.2. Skip 4 Elements (stride = 32 bytes)

```asm
1:
    mov    (%r8), %rax      # READ FROM CACHE L1
    add    $32, %r8         # skip 4 elements
    sub    $1, %rdi
    jg     1b
```

* Still 60 clocks per 32 elements read, but we use only every 4th.
* Effective cost:

  $$
  \frac{60}{8} \approx 7.5 \text{ clocks per element (used)}
  $$

---

## 5. Larger Skips

### 5.1. Skipping 128 Bytes

```asm
1:
    mov    (%r8), %rax      # READ FROM CACHE L1
    add    $128, %r8        # skip 16 elements (16 * 8 = 128 bytes)
    sub    $1, %rdi
    jg     1b
```

* CAS + 15 = 60 clocks read 32 elements in the simplified burst model.
* We only use 2 elements out of that pattern.
* Effective cost:

  $$
  \frac{60}{2} \approx 30 \text{ clocks per element}
  $$

### 5.2. Skipping 256 Bytes

```asm
1:
    mov    (%r8), %rax      # READ FROM CACHE L1
    add    $256, %r8        # skip 32 elements
    sub    $1, %rdi
    jg     1b
```

* Same 60-clock burst model, but effectively **one useful element per burst**.
* Effective cost:

  $$
  \frac{60}{1} \approx 60 \text{ clocks per element}
  $$

(Simplified: you’re now essentially paying a full burst per element.)

---

## 6. Very Large Skips (New Rows/Pages)

### Skipping 8192 Bytes (going to a different row/page)

```asm
1:
    mov    (%r8), %rax      # READ FROM CACHE L1
    add    $256, %r8        # (as written in the notes; represents a large stride)
    sub    $1, %rdi
    jg     1b
```

For large strides that frequently cross rows/banks:

* You pay **RAS + CAS + burst** each time:

  $$
  \text{RAS} + \text{CAS} + 15 = 46 + 45 + 15 = 106 \text{ clocks}
  $$

* If only **one element** is effectively used per burst:

  $$
  \frac{106}{1} \approx 106 \text{ clocks per element}
  $$

---

## Takeaways

* **Sequential access**:
	* Takes advantage of cache lines and DRAM bursts.
	* Minimizes RAS/CAS overhead per element.
* **Strided / skipping access**:
	* Wastes much of what is fetched in each burst.
	* Quickly increases cost per useful element.
* **Random / large-stride access**:
	* Often forces new row/bank activations.
	* Extremely high latency per element (tens to hundreds of cycles).

These rules motivate:

* **Contiguous data structures**.
* Cache-friendly iteration patterns.
* Avoiding large, irregular strides in performance-critical code.
