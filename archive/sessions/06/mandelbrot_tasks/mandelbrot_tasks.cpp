/*
 * Copyright (c) 2010-2023, Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#pragma warning(disable : 4244)
#pragma warning(disable : 4305)
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "mandelbrot_ispc.h"
#include "timing.h"

using namespace std;
using namespace ispc;

extern void mandelbrot_serial(float x0, float y0, float x1, float y1, int width, int height, int maxIterations,
							  int output[]);

// Write a PPM image file with the image of the Mandelbrot set
static void writePPM(const int* buf, const int width, const int height, const char* fn) {
	const auto fp = fopen(fn, "wb");
	if (!fp) {
		printf("Couldn't open a file '%s'\n", fn);
		exit(1);
	}
	fprintf(fp, "P6\n");
	fprintf(fp, "%d %d\n", width, height);
	fprintf(fp, "255\n");
	for (auto i = 0; i < width * height; ++i) {
		// Map the iteration count to colors by just alternating between two greys.
		const char c = buf[i] & 0x1 ? static_cast<char>(240) : 20;
		for (auto j = 0; j < 3; ++j)
			fputc(c, fp);
	}
	fclose(fp);
	printf("Wrote image file %s\n", fn);
}

static void usage() {
	fprintf(stderr, "usage: mandelbrot [--scale=<factor>] [tasks iterations] [serial iterations]\n");
	exit(1);
}

int main(const int argc, char* argv[]) {
	static unsigned int test_iterations[] = {7, 1};
	auto width = 1536L;
	auto height = 1024L;
	constexpr auto x0 = -2.0;
	constexpr auto x1 = 1.0;
	constexpr auto y0 = -1.0;
	constexpr auto y1 = 1.0;
	if (argc > 1 && strncmp(argv[1], "--scale=", 8) == 0) {
		const auto scale = strtol(argv[1] + 8, nullptr, 10);
		if (scale == 0.0f)
			usage();
		width *= scale;
		height *= scale;
		// round up to multiples of 16
		width = width + 0xf & ~0xf;
		height = height + 0xf & ~0xf;
	}
	if (argc == 3 || argc == 4)
		for (auto i = 0; i < 2; i++)
			test_iterations[i] = strtol(argv[argc - 2 + i], nullptr, 10);
	constexpr auto maxIterations = 512;
	const auto buf = new int[width * height];
	// Compute the image using the ISPC implementation; report the minimum time of three runs.
	auto minISPC = 1e30;
	for (auto i = 0U; i < test_iterations[0]; ++i) {
		// Clear out the buffer
		for (auto j = 0; j < width * height; ++j)
			buf[j] = 0;
		reset_and_start_timer();
		mandelbrot_ispc(x0, y0, x1, y1, width, height, maxIterations, buf);
		auto dt = get_elapsed_mcycles();
		printf("@time of ISPC + TASKS run:\t\t\t[%.3f] million cycles\n", dt);
		minISPC = min(minISPC, dt);
	}
	printf("[mandelbrot ispc+tasks]:\t[%.3f] million cycles\n", minISPC);
	writePPM(buf, width, height, "mandelbrot-ispc.ppm");
	// And run the serial implementation 3 times, again reporting the minimum time.
	auto minSerial = 1e30;
	for (auto i = 0U; i < test_iterations[1]; ++i) {
		// Clear out the buffer
		for (auto j = 0; j < width * height; ++j)
			buf[j] = 0;
		reset_and_start_timer();
		mandelbrot_serial(x0, y0, x1, y1, width, height, maxIterations, buf);
		auto dt = get_elapsed_mcycles();
		printf("@time of serial run:\t\t\t[%.3f] million cycles\n", dt);
		minSerial = min(minSerial, dt);
	}
	printf("[mandelbrot serial]:\t\t[%.3f] million cycles\n", minSerial);
	writePPM(buf, width, height, "mandelbrot-serial.ppm");
	printf("\t\t\t\t(%.2fx speedup from ISPC + tasks)\n", minSerial / minISPC);
	return 0;
}
