/*
	Author: Dov Kruger
	Cleaned up, unified Mandelbrot demo combining
	gangs (vectorized code)
	tasks (multithreaded code)
	taken from Intel Demo
*/

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#pragma warning(disable : 4244)
#pragma warning(disable : 4305)
#endif

// since this is pulled out, we have to copy the timing utility
#include <cstdio> //TODO: remove, this is only used for awful Intel writeppm function, replace with better one
#include <iomanip>
#include <iostream>

#include "mandelbrot_ispc.h"
#include "mandelbrot_tasks_ispc.h"
#include "timing.h"

using namespace std;
using namespace ispc;

// The single-threaded, normal code for mandelbrot
extern void mandelbrot_serial(float x0, float y0, float x1, float y1, int width, int height, int maxIterations,
							  int output[]);

// TODO: replace this with color and .webp or png
/* Write a PPM image file with the image of the Mandelbrot set */
static void writePPM(const uint32_t* buf, const int width, const int height, const char* fn) {
	FILE* fp = fopen(fn, "wb");
	if (!fp) {
		printf("Couldn't open a file '%s'\n", fn);
		exit(-1);
	}
	fprintf(fp, "P6\n");
	fprintf(fp, "%d %d\n", width, height);
	fprintf(fp, "255\n");
	for (auto i = 0; i < width * height; ++i) {
		// Map the iteration count to colors by just alternating between
		// two greys.
		const char c = buf[i] & 0x1 ? static_cast<char>(240) : 20;
		for (auto j = 0; j < 3; ++j)
			fputc(c, fp);
	}
	fclose(fp);
	printf("Wrote image file %s\n", fn);
}

// Execute function f, benchmark it a number of times, and print the best time
template <typename Func>
void benchmark1(const uint32_t num_trials, const char msg[], uint32_t max_iterations, float x0, float x1, float y0,
				float y1, uint32_t width, uint32_t height, uint32_t counts[], Func f) {
	auto min_time = 1e100;
	cout << msg << endl;
	for (unsigned int trials = 0; trials < num_trials; ++trials) {
		reset_and_start_timer();
		f(x0, y0, x1, y1, width, height, max_iterations, reinterpret_cast<int*>(counts));
		double dt = get_elapsed_mcycles();
		cout << setprecision(3) << dt << '\t';
		min_time = min(min_time, dt);
	}
	cout << "\nBest time: " << min_time << endl;
}

/*
	verify that the parallel and scalar versions work the same
	 by comparing the numbers
	 TODO: should print how many errors, not each one but probably everything is fine so not bothering
 */
void verify(uint32_t counts1[], uint32_t counts2[], const uint32_t n) {
	for (uint32_t i = 0; i < n; i++)
		if (counts1[i] != counts2[i])
			cout << "Error at " << i << endl;
}

void bench(const uint32_t num_trials, const uint32_t vector_size, uint32_t res, const float x0, const float x1,
		   const float y0, const float y1, const uint32_t max_iterations) {
	if (res % vector_size != 0) {
		res = (res + vector_size) % vector_size; // round up to next even multiple
		const float aspect_ratio = (x1 - x0) / (y1 - y0);
		// right now this is 3:2, and this has to be made robust. We need a multiple of 8 for avx2, 16 for avx512
		const uint32_t width = res * aspect_ratio, height = res;
		cout << "benchmarking Mandelbrot width=" << width << ", height=" << height << endl;
		// allocate integers to compute the images
		const uint32_t num_pixels = width * height;
		const auto counts_serial = new uint32_t[num_pixels];
		const auto counts_parallel = new uint32_t[num_pixels];
		// TODO: note order of parameters is different not changing for now
		// TODO: very sloppy intel!
		benchmark1(num_trials, "scalar", x0, x1, y0, y1, width, height, max_iterations, counts_serial,
				   mandelbrot_serial);
		benchmark1(num_trials, "vectorized", x0, x1, y0, y1, width, height, max_iterations, counts_parallel,
				   mandelbrot_ispc);
		verify(counts_serial, counts_parallel, num_pixels);

		// Clear out the buffer to make sure the parallel vectorized test is really getting results...
		for (uint32_t i = 0; i < num_pixels; ++i)
			counts_parallel[i] = 0;

		// this one is both multithreaded and vectorized
		benchmark1(num_trials, "parallel", x0, x1, y0, y1, width, height, max_iterations, counts_parallel,
				   mandelbrot_tasks_ispc);
		verify(counts_serial, counts_parallel, num_pixels);

		// write out one image since they are all the same
		writePPM(counts_parallel, width, height, "mandelbrot.ppm");

		delete[] counts_serial;
		delete[] counts_parallel;
	}
}


int main(int argc, char* argv[]) {
	constexpr uint32_t res = 4096;

	/*
		test will always compute the image with equal pixel sizes in x and y
		generate classic mandelbrot picture 3 ways
		single threaded, scalar
		using vectorized (gangs in ispc)
		using multiple cores
		benchmark to show timings and compare the three to make sure results
		are the same
	*/
	{
		constexpr uint32_t num_trials = 5;
		constexpr uint32_t vector_size =
			16; // for AVX512, let's make sure the number works for all CPUs the same for now
		constexpr uint32_t max_iterations = 512; // the multi-task version used this number
		constexpr float x0 = -2, x1 = 1, y0 = -1, y1 = 1;
		bench(num_trials, vector_size, res, x0, x1, y0, y1, max_iterations);
	}

#if 0
		//TODO: Make a movie zooming in?
		// this would create frames mandelmovie0, mandelmovie1, ...
		// turn into a movie using ffmpeg or equivalent
		{
			const float x0 = -2, x1 = 1, y0 = -1, y1 = 1;
			for (int frame = 0; frame < num_frames; frame++) {
				movie(res, x0, x1, y0, y1, "mandelmovie", frame);
				// calculate next x0,x1,y0,y1 and do it again
			}
		}
#endif
	return 0;
}
