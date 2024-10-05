#include <cstdint>
#include <cmath>
#include <complex>
#include <iostream>
#include <fstream>
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
            for (uint32_t count = 0; count < max_count; count++, out++) {
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

void convert_mandelbrot_count_to_rgb(uint32_t pixels[], uint32_t mandelbrot_count[], uint32_t w, uint32_t h, const uint32_t colors[], uint32_t color_count) {
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            uint32_t index = y * w + x;
            uint32_t mandelbrot_count = pixels[index];

            // Normalize the Mandelbrot iteration count and map it to a color
            uint32_t color_index = mandelbrot_count % color_count;  // Cyclic mapping if count > color_count
            pixels[index] = colors[color_index];
        }
    }
}
// codium, you idiot, the colors should range from 0 to 255...
void build_color_table(uint32_t colors[], uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        // Generate a color based on the position in the palette
        uint8_t r = (i * 5) % 256;  // Adjust values to create a gradient
        uint8_t g = (i * 7) % 256;  // Feel free to tweak the multipliers
        uint8_t b = (i * 11) % 256; // to achieve different patterns
        uint8_t a = 0xFF;           // Set transparency to opaque

        // Combine color components into a single 32-bit value
        colors[i] = (a << 24) | (r << 16) | (g << 8) | b;
    }
}


bool save_webp(const char* filename, uint32_t* pixels, uint32_t w, uint32_t h, int quality) {
    // Convert the array of pixels (in RGBA format) to a WebP-encoded buffer
    uint8_t* webp_data;
    size_t webp_size = WebPEncodeRGBA((uint8_t*)pixels, w, h, w * 4, quality, &webp_data);
    
    if (webp_size == 0) {
        std::cerr << "Error encoding WebP image!" << std::endl;
        return false; // Encoding failed
    }

    // Save the WebP-encoded buffer to a file
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing!" << std::endl;
        WebPFree(webp_data); // Free the WebP data in case of error
        return false;
    }

    file.write(reinterpret_cast<const char*>(webp_data), webp_size);
    file.close();
    
    // Free the WebP buffer allocated by WebPEncodeRGBA
    WebPFree(webp_data);

    return true;
}

int main() {
    const int w = 1920, h = 1080;
    uint32_t* mandelbrot_counts = new uint32_t[w*h];
    uint32_t* pixels = new uint32_t[w*h];

    uint32_t colors[64];
    // come on codeium, come up with colors for a nice mandelbrot...
    build_color_table(colors, 64);
    mandelbrot(mandelbrot_counts, w, h, 64, -2, 0.47, -1.12, 1.12);

    convert_mandelbrot_count_to_rgb(pixels, mandelbrot_counts, w, h, colors, 64);


    delete [] mandelbrot_counts;
    delete [] pixels;
}