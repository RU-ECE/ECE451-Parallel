# Midterm Concepts

1. **Multithreading**
2. **Limitations of Memory**
	1. **Sequential access**
		- Accessing memory in order is generally faster due to caching and prefetching.
	2. **Cache hierarchy**
		- **Level 1 (L1)**: very small, very fast
			- Example: 128 KB data cache (D-cache), 128 KB instruction cache (I-cache)
		- **Level 2 (L2)**: bigger, slightly slower
			- Example: 2 MB
		- **Level 3 (L3)**: shared, much larger, slower than L1/L2
			- Example: 32 MB
	3. **Read/Write behavior**
		- Different latencies and potential contention on loads vs. stores.
	4. **Burst mode**
		- Example pattern: first access is slow, following accesses (within the same burst) are fast.
		- Think: 1st word = slow, next 7 = much faster.
	5. **Dual banks**
		- Memory organized in multiple banks to increase effective bandwidth.
		- Example: 2 banks $\Rightarrow$ can service more accesses in parallel (e.g., up to 16 in flight).
	6. **Rows and columns**
		- DRAM is organized like a 2D array (rows and columns).
		- Row activation and row-buffer behavior affect performance.
	7. **Memory manager**
		- **Not on the midterm.**
3. **Pipelining**
	- Breaking execution into stages so multiple instructions are in-flight simultaneously.
	- Hazards: data, control, and structural hazards (and how they are mitigated).
4. **Optimizer (Compiler Optimizations)**
	1. The optimizer can remove code it considers "unnecessary."
		- Dead code elimination, constant folding, etc.
	2. This can **fool you** when:
		- Measuring performance with dummy loops.
		- Trying to benchmark code that has no observable effect.
	3. Use:
		- `volatile`, actual outputs, or side effects to keep important code from being optimized away.
