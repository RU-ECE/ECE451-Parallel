// default is sequential partitions
void product(const float a[], const float b[], float c[], const int n) {
#pragma omp parallel for
	for (auto i = 0; i < n; ++i)
		c[i] = a[i] * b[i];
}

float sum_array(const float arr[], const int n) {
	float sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
	for (auto i = 0; i < n; ++i)
		sum += arr[i];
	return sum;
}

// float = 4 bytes
// double = 8 bytes
// avx2 = 256 bits = 32 bytes
// factor 8 speedup
void multiply_by_scalar(const float a[], const float b, float c[], const int n) {
#pragma omp parallel for simd
	for (auto i = 0; i < n; ++i) // load %Ymm0 = (b, b, b, b, b, b, b, b)
		c[i] = a[i] * b;
}

float mandelbrot_point(const float x, const float y, const int max_iter) {
	float zr = 0.0, zi = 0.0;
	auto iter = 0;
#pragma omp simd
	for (iter = 0; iter < max_iter; ++iter) {
		const float zr2 = zr * zr;
		const float zi2 = zi * zi;
		if (zr2 + zi2 > 4.0)
			break;
		zi = 2.0 * zr * zi + y;
		zr = zr2 - zi2 + x;
	}
	return static_cast<float>(iter) / max_iter;
}


void mandelbrot(const float xmin, const float xmax, const float ymin, const float ymax, const int width,
				const int height, const int max_iter, float* output) {
#pragma omp parallel for
	for (auto i = 0; i < width; ++i) {
		for (auto j = 0; j < height; ++j) {
			output[i * height + j] =
				mandelbrot_point(xmin + i * (xmax - xmin) / width, ymin + j * (ymax - ymin) / height, max_iter);
		}
	}
}

// using avx512, factor 16 speedup

// theoretical = number of threads * number of elements per thread = 4 * 8 = 32
void matrix_multiply(const float A[], const float B[], float C[], const int N) {
#pragma omp parallel for // factor of number of threads
	for (auto i = 0; i < N; ++i) {
		for (auto j = 0; j < N; ++j) {
			float sum = 0.0;
#pragma omp simd // factor of 8
			for (auto k = 0; k < N; ++k)
				sum += A[i * N + k] * B[k * N + j];
			C[i * N + j] = sum;
		}
	}
}

/*
	The reason speed will improve with transpose is:

	matrix multiply is a = n^3 operations
	if we transpose, it costs n^2
	but then we can go sequential for the n^3
*/
