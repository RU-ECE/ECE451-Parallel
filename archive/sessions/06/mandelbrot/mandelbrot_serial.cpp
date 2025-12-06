/*
 * Copyright (c) 2010-2023, Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

static int mandel(const float c_re, const float c_im, const int count) {
	auto z_re = c_re, z_im = c_im;
	auto i = 0;
	while (i < count) {
		if (z_re * z_re + z_im * z_im > 4.0f)
			break;
		z_re = c_re + z_re * z_re - z_im * z_im;
		z_im = c_im + 2 * z_re * z_im;
		i++;
	}
	return i;
}

void mandelbrot_serial(const float x0, const float y0, const float x1, const float y1, const int width,
					   const int height, const int maxIterations, int output[]) {
	for (auto j = 0; j < height; j++) {
		for (auto i = 0; i < width; ++i) {
			output[j * width + i] = mandel(x0 + i * (x1 - x0) / static_cast<float>(width),
										   y0 + j * (y1 - y0) / static_cast<float>(height), maxIterations);
		}
	}
}
