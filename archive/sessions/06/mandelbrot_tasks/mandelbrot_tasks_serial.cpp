/*
 * Copyright (c) 2010-2023, Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

static int mandel(const float c_re, const float c_im, const int count) {
	auto z_re = c_re, z_im = c_im;
	int i;
	for (i = 0; i < count; ++i) {
		if (z_re * z_re + z_im * z_im > 4.0f)
			break;
		const auto new_re = z_re * z_re - z_im * z_im, new_im = 2 * z_re * z_im;
		z_re = c_re + new_re;
		z_im = c_im + new_im;
	}
	return i;
}

void mandelbrot_serial(const float x0, const float y0, const float x1, const float y1, const int width,
					   const int height, const int maxIterations, int output[]) {
	const auto dx = (x1 - x0) / width, dy = (y1 - y0) / height;
	for (auto j = 0; j < height; j++) {
		for (auto i = 0; i < width; ++i) {
			const auto x = x0 + i * dx, y = y0 + j * dy;
			const auto index = j * width + i;
			output[index] = mandel(x, y, maxIterations);
		}
	}
}
