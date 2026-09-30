#include <webp/encode.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

/*
    Mandelbrot set:   z = z^2 + c
    z = (zr, zi)
    z^2 = ((zr^2 - zi^2) , 2*zr*zi)


    abs = sqrt(zr*zr + zi*zi) > 2 it will explode exponentially
    
    zr*zr + zi*zi < 4

    z^2 = zr^2 - xi^2 + yr^2 - yi^2

*/

uint32_t mandel(float c_re, float c_im, int count) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.)
            break;
            // now calculate z^2+c
        float tmp = z_re * z_re - z_im * z_im + c_re;
        z_im = 2. * z_re * z_im + c_im;
        z_re = tmp;
    }
    return i;
}


void mandelbrot(uint32_t w, uint32_t h,
     float x0, float y0, float x1, float y1, uint32_t count, uint32_t output[]) {
    const float dx = (x1 - x0) / w;
    const float dy = (y1 - y0) / h;
    for (uint32_t j = 0; j < h; j++) {
        float y = y0 + j * dy;
        for (uint32_t i = 0; i < w; ++i) {
            float x = x0 + i * dx;
            int index = (j * w + i);
            output[index] = mandel(x, y, maxIterations);
        }
    }
}

#include <webp/encode.h>
#include <cstdio>
#include <cstdint>

bool save(uint32_t* output, uint32_t w, uint32_t h, const char* filename) {
    uint8_t* data = nullptr;

    const size_t size = WebPEncodeRGBA(
        reinterpret_cast<const uint8_t*>(output),
        w, h, // size of the bitmap
        w * sizeof(uint32_t), // size of each row
        90.0f,
        &data);

    if (size == 0)
        return false;

    FILE* f = std::fopen(filename, "wb");
    if (!f) {
        WebPFree(data);
        return false;
    }

    const bool ok = std::fwrite(data, 1, size, f) == size;
    std::fclose(f);
    WebPFree(data);

    return ok;
}

int main() {
    const int w = 2048, h= 2048;
    uint32_t* output = new uint32_t[w*h];
    mandelbrot(w, h, -2.0, -1.0, 1.0, 1.0, output);
    // lookup colors and map
    save(output, w, h, "mandelbrot.webp");
    delete [] output;
    return 0;
}