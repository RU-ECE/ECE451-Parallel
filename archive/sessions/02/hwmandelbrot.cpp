/*
 * Go from (-2,-1) to (2,1)
 *
 * For each point (x,y), calculate the number of iterations until |Z| > 2
 */

constexpr auto MAX_ITERATIONS = 100;

void mandelbrot(unsigned int pixels[], const int width, const int height) {
	for (auto j = 0; j < height; j++) {
		const auto y = 2.0f * j / height - 1;
		for (auto i = 0; i < width; i++) {
			// C = (x,y)
			auto zx = 0.0;
			constexpr auto zy = 0.0;
			int count;
			for (count = 0; count < MAX_ITERATIONS && zx * zx + zy * zy < 4; count++) {
				const auto z_temp = zx * zx + zy * zy; // Z = Z * Z + C;
				const float z_y = zx * z_y * 2 + y;
				zx = z_temp;
			}
			pixels[j * width + i] = count;
		}
	}
}

// save out the picture!
// https://solarianprogrammer.com/2019/06/10/c-programming-reading-writing-images-stb_image-libraries/
