#include <cmath>
#include <complex>

using namespace std;

void mandel(unsigned int rgba[], const unsigned int w, const unsigned int h, const unsigned long max_iter,
			const float xmin, const float xmax, const float ymin, const float ymax) {
	const auto dx = (xmax - xmin) / (w - 1), dy = (ymax - ymin) / (h - 1);
	for (auto i = 0U; i < h; i++) {
		const auto y = ymin + dy * i;
		for (auto j = 0U; j < w; j++) {
			const auto x = xmin + dx * j;
			// C =(x, y)
			// Z = C
			// f(Z) = Z^2 + C
			complex c(x, y);
			auto z = c;
			for (auto k = 0U; k < max_iter && abs(z) < 2; k++)
				z = z * z + c;
			// write the count to the pixels
		}
	}

	// now convert count to colors
}
