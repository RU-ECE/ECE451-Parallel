# CUDA

## OpenGL legacy

OpenGL started out simply

1. CPU makes function calls to draw to graphics card (slow)
2. Graphics card has independent functions (hardcoded). CPU sends data to graphics card, graphics card does the work (COMPLICATED)
3. Graphics card becomes general purpose computer
  * shaders for graphics
  * compute shaders
  * CUDA is a compute API for graphics
  * transfer data back and forth

Potential problems
1. Is it slower to send the data to the graphics card than to just compute it?
2. Can we get more net computing done if we do this?





Today: OpenGL supports vertex shaders, fragment shaders, and compute shaders

Vulkan = new OpenGL for C++ (multithreaded, object-oriented, extremely complicated, faster)

CUDA (NVIDIA's language for parallel coding)

