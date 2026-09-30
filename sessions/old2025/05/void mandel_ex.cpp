#include <cmath>
void mandel(uint32_t rgba[], uint32_t w, uint32_t h, 
   uint64_t max_iter,
    float xmin, float xmax, float ymin, float ymax) {
  const double dx = (xmax-xmin) / (w-1), 
            dy = (ymax - ymin) / (h-1);
  for (uint32_t i = 0; i < h; i++) {
    const float y = ymin + dy*i;
    for (uint32_t j = 0; j < w; j++) {
        float x = xmin + dx*j;
        // C =(x, y)
        // Z = C
        // f(Z) = Z^2 + C
        complex c(x,y);
        complex z = c;
        uint32_t i;
        for (i = 0; i < max_iter && abs(z) < 2; i++) {
            z = z*z + c;
        }
        // write the count to the pixels
    }
  }

  // now convert count to colors
}