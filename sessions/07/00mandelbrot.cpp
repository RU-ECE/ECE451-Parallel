/*
    C = (0,0)
    f(z) = z^2 + C 
            (0,0) + (0,0)

    C = (2,0)
    f(z) = z^2 + C
            (4,0) + (2,0) = (6,0)
            (36,0) + (6,0)...
*/
#include <iostream>
#include <cstdint>
#include <cmath>
#include <complex>

using namespace std;
void mandelbrot(uint32_t count_arr[], uint32_t w, uint32_t h,
                const uint32_t max_count,
                const float xmin, const float xmax,
                const float ymin, const float ymax){
    int out = 0; // sequentially write each count to array
    for (uint32_t i = 0; i < h; i++) {
        float y = ymin + (ymax-ymin)*i/h;
        for (uint32_t j = 0; j < w; j++) {
            float x = xmin + (xmax-xmin)*j/w;
            complex c(x,y);
            complex z = c;
            for (uint32_t count = 0; count < max_count; count++) {
                z = z*z + c; // C++ operator overload
                if (abs(z) > 2) {
                  count_arr[out] = count; 
                  break;
                }
            }
            count_arr[out] = max_count;
        }
    }
}