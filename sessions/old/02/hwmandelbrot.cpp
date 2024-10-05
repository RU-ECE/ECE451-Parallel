#include <immintrin.h>
/*
    go from (-2,-1) to (2,1)
    for each point (x,y) calculate the number of iterations until |Z| > 2

*/
const int MAX_ITERATIONS = 100;
void mandelbrot(uint32_t pixels[], int width, int height) {
    for (int j = 0; j < height; j++) {
        float y = 2 * j / (float)height - 1;
        for (int i = 0; i < width; i++) {
            float x = 2 * i / (float)width - 1;
            // C = (x,y)
            float zx = 0;
            float zy = 0;
            int count;
            for (count = 0; count < MAX_ITERATIONS && zx*zx + zy*zy < 4; count++) {
                float ztemp = zx * zx + zy * zy; //            Z = Z * Z + C;
                float zy = zx * zy * 2 + y;
                zx = ztemp;
            }

            pixels[j * width + i] = count;

        }
    }
}

// save out the picture!
//https://solarianprogrammer.com/2019/06/10/c-programming-reading-writing-images-stb_image-libraries/