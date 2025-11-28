/*
 * Copyright (c) 2010-2023, Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

static int mandel(const float c_re, const float c_im, const int count) {
	float z_re = c_re, z_im = c_im;
	int i;
	for (i = 0; i < count; ++i) {
		if (z_re * z_re + z_im * z_im > 4.f)
			break;

		const float new_re = z_re * z_re - z_im * z_im;
		const float new_im = 2.f * z_re * z_im;
		z_re = c_re + new_re;
		z_im = c_im + new_im;
	}

	return i;
}

void mandelbrot_serial(const float x0, const float y0, const float x1, const float y1, const int width,
					   const int height, const int maxIterations, int output[]) {
	const float dx = (x1 - x0) / width;
	const float dy = (y1 - y0) / height;

	for (auto j = 0; j < height; j++) {
		for (auto i = 0; i < width; ++i) {
			const float x = x0 + i * dx;
			const float y = y0 + j * dy;

			const int index = j * width + i;
			output[index] = mandel(x, y, maxIterations);
		}
	}
}
