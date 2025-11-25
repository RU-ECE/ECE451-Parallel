# HW Mandelbrot GPU

Write code to generate the image of the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set)

We wrote some code in class demonstrating Mandelbrot. You may use it, or any other code as a guide. Your group is reponsible to write your own code. As usual you must cite any help or sources

1. Write a GPU-based mandelbrot in either CUDA or HIP that calculates for each element how many times it took for the point to escape:

```cpp
void mandelbrot(uint32_t counts[], uint32_t maxcount, uint32_t w, uint32_t h, float xmin, float xmax, float ymin, float ymax);

//The test for this homework will be
mandelbrot(counts, 256, 1280,1024, -1, 1, -1, 1)
```
Compare the speed of your GPU solution to the scalar CPU solution. Record the model GPU used to run the test.
Results should be pasted in the comment at the top of the program below your name and any citations.

2. Generate a picture of the mandelbrot. You may do so by looking up a color based on the number and either saving to a .webp file (batch, good for Amarel) or using OpenGL code provided which will allow you to draw dynamically in a window. see [web docs](https://developers.google.com/speed/webp/docs/api)
Be aware that getting OpenGL to work may take some installing of various software libraries. 
