# Homework: Mandelbrot GPU

In this assignment, you will write GPU code to generate an image of
the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set).

We wrote some Mandelbrot code in class as a demonstration. You may use that, or any other reference code, **as a guide
only**. Your group is responsible for writing your **own** implementation. As usual, you must cite any help or external
sources you use.

---

## 1. GPU Mandelbrot Kernel

Write a **GPU-based** Mandelbrot implementation in either **CUDA** or **HIP** that computes, for each pixel, how many
iterations it takes for the point to escape.

Use the following function prototype:

```c++
void mandelbrot(uint32_t counts[],
                uint32_t maxcount,
                uint32_t w,
                uint32_t h,
                float xmin,
                float xmax,
                float ymin,
                float ymax);
````

The test that will be used for this homework is:

```c++
mandelbrot(counts, 256, 1280, 1024, -1, 1, -1, 1);
```

### 1.1. Performance Comparison

* Implement a **scalar CPU version** of Mandelbrot.
* Implement your **GPU version** in CUDA or HIP.
* Compare the **runtime** of:
	* The scalar CPU implementation, and
	* Your GPU implementation.

Record:

* The **GPU model** used (for example, RTX 3080, MI250, etc.).
* The **timing results** for both CPU and GPU versions.

These results should be pasted in a **comment at the top of your program**, underneath:

* Your name(s), and
* Any citations.

---

## 2. Generating a Mandelbrot Image

Generate a visual image of the Mandelbrot set using your computed iteration counts.

You may:

1. Map the iteration count for each pixel to a **color** (for example, via a colormap or simple gradient), and
2. Output the result as an image using **one** of the following approaches:

### Option A: Save to `.webp` File (Batch Mode)

* Use a library that can write `.webp` files (good for batch processing or running on Amarel).
* See the WebP API documentation:
  [https://developers.google.com/speed/webp/docs/api](https://developers.google.com/speed/webp/docs/api)

### Option B: Use OpenGL for Interactive Display

* Use the provided **OpenGL code** to visualize the Mandelbrot set in a window.
* This allows dynamic drawing and interaction.
* Be aware that getting OpenGL working may require installing additional software libraries on your system.
