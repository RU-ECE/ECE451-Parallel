# Potential Algorithms for CUDA and ISCD/OpenMP

Some candidate algorithms you could implement and parallelize:

1. **Mandelbrot / Fractals**
	- 2D Mandelbrot set
	- 3D variants such as *Mandelbulber*-style fractals

2. **Merge Sort**
	- Overall algorithm is parallel at each merge *level*.
	- However, merging **two** sorted lists into one is mostly **sequential**, so this is **not ideal for CUDA** (fine for CPU OpenMP-style work, less good for massively parallel GPUs).

3. **Matrix Algorithms**
	- Matrix multiplication
	- Gram–Schmidt orthogonalization
	- Solving systems of linear equations
	- Computing a spline for $n$ points

4. **Graph Theory**
	- High-speed **CSR (Compressed Sparse Row)** implementations
	- Parallel operations on sparse graphs (BFS, PageRank, etc.)

5. **FFT**
	- Fast Fourier Transform implementations
	- Good candidate for GPU or multicore parallelism
