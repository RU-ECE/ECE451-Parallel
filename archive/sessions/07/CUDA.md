# CUDA

## OpenGL Background (Legacy to Now)

OpenGL started out fairly simple and evolved over time:

1. **Stage 1: Immediate Mode (Very Early OpenGL)**
	- The **CPU** makes function calls directly to draw to the graphics card.
	- Example: `glBegin()`, `glVertex()`, `glEnd()`.
	- Simple, but **slow** and not very flexible.

2. **Stage 2: Fixed-Function Pipeline**
	- The graphics card has **hardcoded**, fixed functions:
		- Transformations
		- Lighting
		- Texturing
	- The CPU sends data (vertices, textures, etc.) to the GPU.
	- The GPU performs the fixed sequence of operations.
	- Much faster, but:
		- Behavior is **limited**.
		- Pipeline is **complicated** to configure.

3. **Stage 3: Programmable Pipeline**
	- The GPU becomes a **general-purpose parallel processor**.
	- Introduces:
		- **Vertex shaders** (programs run per vertex)
		- **Fragment shaders** (programs run per pixel/fragment)
		- Later: **compute shaders** (general computation, not just graphics).
	- We now:
		- Upload data to GPU.
		- Run custom shader programs.
		- Optionally transfer results back to CPU.

---

## CUDA and Compute on the GPU

- **Shaders for graphics**:
	- Vertex, fragment, geometry, etc.
- **Compute shaders**:
	- General-purpose code running on the GPU through the graphics API.
- **CUDA**:
	- NVIDIA’s **compute API and language** for general-purpose GPU (GPGPU) programming.
	- Does not require going through the graphics pipeline.
	- Lets you:
		- Allocate GPU memory.
		- Launch many parallel threads (kernels).
		- Transfer data back and forth between CPU (host) and GPU (device).

---

## Potential Problems / Tradeoffs

When using the GPU for general computation (via CUDA or compute shaders), consider:

1. **Data Transfer Overhead**
	- Is it **slower to send the data to the GPU** than to just compute it on the CPU?
	- PCIe transfer can be expensive for small problems.

2. **Net Computation Gain**
	- Even with transfer costs, can we get **more total work done** by using the GPU’s massive parallelism?
	- GPU is great for **large, highly parallel workloads** where computation dominates transfer time.

---

## Modern APIs

- **OpenGL (Today)** supports:
	- Vertex shaders
	- Fragment (pixel) shaders
	- Compute shaders

- **Vulkan**
	- A modern, low-level graphics and compute API.
	- Designed for C++:
		- Multithreaded
		- Object-oriented style
		- Very explicit and **extremely complicated**, but **faster** and more controllable.

- **CUDA**
	- NVIDIA’s language and runtime for **parallel coding on NVIDIA GPUs**.
	- Focused on **compute**, not on drawing directly to the screen (though it can interoperate with OpenGL/Vulkan).
