__device__ void order(float& a, float& b) { // theoretically should be a register
	if (a > b) { // but pass by reference is not good, this does not work on CPU
		const float temp = a;
		a = b;
		b = temp;
	}
}

/*
	first load each group of 8 elements into registers
	then sort each group of 8 elements using the order function
	should be optimal unless the CUDA compiler doesn't deal with references well
	but that's not theoretically our fault, we will consider that this works
	At the end, write each group of 8 elements back to the global memory
*/
__global__ void sort8(float arr[], const int n) {
	for (auto i = 0; i < n; i += 8) {
		float a = arr[i]; // global memort on GPU
		float b = arr[i + 1];
		float c = arr[i + 2];
		float d = arr[i + 3];
		float e = arr[i + 4];
		float f = arr[i + 5];
		float g = arr[i + 6];
		float h = arr[i + 7];

		// Initial pairwise comparisons
		order(a, b); // (0, 1)
		order(c, d); // (2, 3)
		order(e, f); // (4, 5)
		order(g, h); // (6, 7)

		// Second layer
		order(a, c); // (0, 2)
		order(b, d); // (1, 3)
		order(e, g); // (4, 6)
		order(f, h); // (5, 7)

		// Third layer
		order(a, e); // (0, 4)
		order(b, f); // (1, 5)
		order(c, g); // (2, 6)
		order(d, h); // (3, 7)

		// Fourth layer
		order(b, c); // (1, 2)
		order(d, e); // (3, 4)
		order(f, g); // (5, 6)

		// Fifth layer
		order(a, b); // (0, 1)
		order(c, d); // (2, 3)
		order(e, f); // (4, 5)
		order(g, h); // (6, 7)

		// now write back to global memory
		arr[i] = a;
		arr[i + 1] = b;
		arr[i + 2] = c;
		arr[i + 3] = d;
		arr[i + 4] = e;
		arr[i + 5] = f;
		arr[i + 6] = g;
		arr[i + 7] = h;
	}
}

// merge groups of 8 elements using shared memory
__global__ void fast_merge_early_passes(float arr[], int n) {
	__shared__ float shared_a[1024];
	__shared__ float shared_b[1024];
	// using shared memory is a lot faster than global memory
	// also each block has it own shared memory
}

/*
	What's wrong with this code? Can you spot the inefficiencies due to memory?
*/
__global__ void dft(const float* arr, float* real_out, float* imag_out, const int n) {
	const int k = blockIdx.x * blockDim.x + threadIdx.x; // Index for the output element
	if (k < n) {
		float real_sum = 0.0;
		float imag_sum = 0.0;
		for (auto t = 0; t < n; ++t) {
			const float angle = 2.0 * M_PI * t * k / n;
			real_sum += arr[t] * cos(angle);
			imag_sum += arr[t] * -sin(angle);
		}
		real_out[k] = real_sum;
		imag_out[k] = imag_sum;
	}
}

// Is this better?
__global__ void dft_with_precomputation(const float* arr, float* real_out, float* imag_out, const int n) {
	extern __shared__ float shared_mem[]; // Shared memory for precomputed values
	float* cos_vals = shared_mem;
	float* sin_vals = shared_mem + n;

	if (const int t = threadIdx.x; t < n) {
		const float angle = 2.0 * M_PI * t / n;
		cos_vals[t] = cos(angle);
		sin_vals[t] = -sin(angle);
	}
	__syncthreads(); // Ensure all threads have computed the values

	const int k = blockIdx.x * blockDim.x + threadIdx.x; // Index for the output element
	if (k < n) {
		float real_sum = 0.0;
		float imag_sum = 0.0;
		for (auto t = 0; t < n; ++t) {
			real_sum += arr[t] * cos_vals[t * k % n];
			imag_sum += arr[t] * sin_vals[t * k % n];
		}
		real_out[k] = real_sum;
		imag_out[k] = imag_sum;
	}
}

__global__ void fast_dft_into_registers(const float* arr, float* real_out, float* imag_out, const int n) {
	const int t = threadIdx.x;
	if (const int k = blockIdx.x * blockDim.x + t; k < n) {
		float real_sum = 0.0;
		float imag_sum = 0.0;
	}
	// would be faster for custom hand-written code for blocks of 8.
	// would not scale well to larger blocks
	float a = arr[t];
	float b = arr[t + 1];
	float c = arr[t + 2];
	float d = arr[t + 3];
	float e = arr[t + 4];
	float f = arr[t + 5];
	float g = arr[t + 6];
	float h = arr[t + 7];
}

__global__ void dft_shared_memory(const float* arr, float* real_out, float* imag_out, const int n) {
	__shared__ float shared_arr[1024]; // Shared memory for input elements

	const int t = threadIdx.x;
	const int k = blockIdx.x * blockDim.x + t; // Index for the output element

	// Load the first 1024 elements into shared memory
	// FAST AND SEQUENTIAL (EVEN THOUGH IT DOESN'T LOOK LIKE IT)
	if (t < 1024 && t < n)
		shared_arr[t] = arr[t];
	__syncthreads(); // Ensure all threads have loaded the data

	if (k < n) {
		float real_sum = 0.0;
		float imag_sum = 0.0;
		for (auto t = 0; t < 1024 && t < n; ++t) {
			const float angle = 2.0 * M_PI * t * k / n;
			real_sum += shared_arr[t] * cos(angle);
			imag_sum += shared_arr[t] * -sin(angle);
		}
		real_out[k] = real_sum;
		imag_out[k] = imag_sum;
	}
}

__global__ void dft_optimized(const float* arr, float* real_out, float* imag_out, const int n) {
	__shared__ float shared_arr[1024]; // Shared memory for input elements
	__shared__ float shared_real[1024]; // Shared memory for real part of output
	__shared__ float shared_imag[1024]; // Shared memory for imaginary part of output

	const int t = threadIdx.x;
	const int k = blockIdx.x * blockDim.x + t; // Index for the output element

	// Load the first 1024 elements into shared memory
	if (t < 1024 && t < n)
		shared_arr[t] = arr[t];
	__syncthreads(); // Ensure all threads have loaded the data

	if (k < 1024 && k < n) {
		float real_sum = 0.0;
		float imag_sum = 0.0;
		for (auto t = 0; t < 1024 && t < n; ++t) {
			const float angle = 2.0 * M_PI * t * k / n;
			real_sum += shared_arr[t] * cos(angle);
			imag_sum += shared_arr[t] * -sin(angle);
		}
		shared_real[k] = real_sum;
		shared_imag[k] = imag_sum;
	}
	__syncthreads(); // Ensure all threads have completed their calculations

	// Write results from shared memory back to global memory
	if (t < 1024 && t < n) {
		real_out[t] = shared_real[t];
		imag_out[t] = shared_imag[t];
	}

	// Handle offsets larger than 1024 directly using global memory
	if (k >= 1024 && k < n) {
		float real_sum = 0.0;
		float imag_sum = 0.0;
		for (auto t = 0; t < n; ++t) {
			const float angle = 2.0 * M_PI * t * k / n;
			real_sum += arr[t] * cos(angle);
			imag_sum += arr[t] * -sin(angle);
		}
		real_out[k] = real_sum;
		imag_out[k] = imag_sum;
	}
}


int main() { dft<<<1, 256>>>(arr, real_out, imag_out, n); }
