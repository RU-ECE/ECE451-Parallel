# What Would Parallel AVX Sort Code Look Like?

High-level structure of an AVX/AVX-512–based sorting algorithm:

1. **Load as many values as you can into vector registers.**
	- Example: use 8 vector registers (as in the Peloton work).
	- For extra credit, use **AVX-512** with up to 32 512-bit registers.
		- Example design: use 16 registers with about 47 compare–swap stages.
		- Rough instruction count:
			- Swaps: $47 \times 3$ instructions (compare, blend/move, etc.)
			- Transpose: about $120$ instructions
			- Total: $47 \times 3 + 120 \approx 270$ instructions to sort 256 elements.

2. **Sort “vertically” within registers.**
	- Treat each register as a vector of lanes.
	- Apply a sorting network (for example, bitonic or odd–even) using vector compare–swap operations across lanes.

3. **Transpose the data.**
	- Rearrange elements so that what used to be “columns” become “rows” (or vice versa).
	- After the transpose, each register holds a **contiguous** subset of the sorted data.

4. **Now you have registers, each of which is sorted.**
	- Each SIMD register represents a sorted run.

5. **Merge the sorted registers.**
	- Use a parallel merge network (again with vector compares and blends).
	- Continue merging until you obtain one fully sorted sequence in memory.
