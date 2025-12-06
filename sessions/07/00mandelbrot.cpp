#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <webp/encode.h>

/*
	C = (0,0)
	f(z) = z^2 + C
			(0,0) + (0,0)

	C = (2,0)
	f(z) = z^2 + C
			(4,0) + (2,0) = (6,0)
			(36,0) + (6,0)...


			with vecorized coding, if statements are a problem

			because all numbers in the vector are processed the same
*/

using namespace std;

void mandelbrot(unsigned int count_arr[], const unsigned int w, const unsigned int h, const unsigned int max_count,
				const float xmin, const float xmax, const float ymin, const float ymax) {
	auto out = 0; // sequentially write each count to array
	for (auto i = 0U; i < h; i++) {
		const auto y = ymin + (ymax - ymin) * static_cast<float>(i) / static_cast<float>(h);
		for (auto j = 0U; j < w; j++) {
			const auto x = xmin + (xmax - xmin) * static_cast<float>(j) / static_cast<float>(w);
			complex c(x, y);
			complex z = c;
			for (auto count = 0U; count < max_count; count++, out++) {
				z = z * z + c; // C++ operator overload
				if (abs(z) > 2) {
					count_arr[out] = count;
					break;
				}
			}
			count_arr[out] = max_count;
		}
	}
}

void convert_mandelbrot_count_to_rgb(unsigned int pixels[], unsigned int mandelbrot_count[], const unsigned int w,
									 const unsigned int h, const unsigned int colors[],
									 const unsigned int color_count) {
	for (auto y = 0U; y < h; y++) {
		for (auto x = 0U; x < w; x++) {
			const auto index = y * w + x, mandelbrotCount = pixels[index],
					   // Normalize the Mandelbrot iteration count and map it to a color
				color_index = mandelbrotCount % color_count; // Cyclic mapping if count > color_count
			pixels[index] = colors[color_index];
		}
	}
}
// codium, you idiot, the colors should range from 0 to 255...
void build_color_table(unsigned int colors[], const unsigned int count) {
	for (auto i = 0U; i < count; i++) {
		// Generate a color based on the position in the palette
		const auto r = i * 5 % 256; // Adjust values to create a gradient
		const auto g = i * 7 % 256; // Feel free to tweak the multipliers
		const auto b = i * 11 % 256; // to achieve different patterns
		constexpr auto a = 0xFF; // Set transparency to opaque

		// Combine color components into a single 32-bit value
		colors[i] = a << 24 | r << 16 | g << 8 | b;
	}
}

bool save_webp(const char* filename, unsigned int* pixels, const unsigned int w, const unsigned int h,
			   const int quality) {
	// Convert the array of pixels (in RGBA format) to a WebP-encoded buffer
	uint8_t* webp_data;
	const size_t webp_size =
		WebPEncodeRGBA(reinterpret_cast<uint8_t*>(pixels), static_cast<int>(w), static_cast<int>(h),
					   static_cast<int>(w) * 4, static_cast<float>(quality), &webp_data);

	if (webp_size == 0) {
		cerr << "Error encoding WebP image!" << endl;
		return false; // Encoding failed
	}

	// Save the WebP-encoded buffer to a file
	ofstream file(filename, ios::binary);
	if (!file) {
		cerr << "Error opening file for writing!" << endl;
		WebPFree(webp_data); // Free the WebP data in case of error
		return false;
	}

	file.write(reinterpret_cast<const char*>(webp_data), static_cast<long>(webp_size));
	file.close();

	// Free the WebP buffer allocated by WebPEncodeRGBA
	WebPFree(webp_data);

	return true;
}

int main() {
	constexpr auto w = 1920, h = 1080;
	const auto mandelbrot_counts = new unsigned int[w * h];
	const auto pixels = new unsigned int[w * h];
	unsigned int colors[64];
	// come on codeium, come up with colors for a nice mandelbrot...
	build_color_table(colors, 64);
	mandelbrot(mandelbrot_counts, w, h, 64, -2, 0.47, -1.12, 1.12);
	convert_mandelbrot_count_to_rgb(pixels, mandelbrot_counts, w, h, colors, 64);
	delete[] mandelbrot_counts;
	delete[] pixels;
}
