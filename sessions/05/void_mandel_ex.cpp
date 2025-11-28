#include <cmath>
#include <complex.h>
#include <cstdint>

using namespace std;

void mandel(uint32_t rgba[], const uint32_t w, const uint32_t h, const uint64_t max_iter, const float xmin,
			const float xmax, const float ymin, const float ymax) {
	const double dx = (xmax - xmin) / (w - 1), dy = (ymax - ymin) / (h - 1);
	for (uint32_t i = 0; i < h; i++) {
		const float y = ymin + dy * i;
		for (uint32_t j = 0; j < w; j++) {
			const float x = xmin + dx * j;
			// C =(x, y)
			// Z = C
			// f(Z) = Z^2 + C
			complex c(x, y);
			auto z = c;
			for (uint32_t i = 0; i < max_iter && abs(z) < 2; i++)
				z = z * z + c;
			// write the count to the pixels
		}
	}

	// now convert count to colors
}
