cmake_minimum_required(VERSION 3.22)

# Enable CUDA as a first-class language (for .cu files)
project(Rutgers_ECE451_Parallel LANGUAGES CXX CUDA)

# C++23 for all C++ sources
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Optional: nicer warnings in GCC/Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# --- Find dependencies (best-effort; warn if missing)
find_package(OpenMP)            # OpenMP::OpenMP_CXX
find_package(MPI)               # MPI::MPI_CXX
find_package(Threads REQUIRED)  # Threads::Threads (for std::thread)
find_package(CUDAToolkit)       # CUDA::cudart (if Toolkit is installed)
find_package(PkgConfig REQUIRED)
pkg_check_modules(WEBP QUIET IMPORTED_TARGET libwebp)              # WebP::webp

# If you want configuration to fail hard when some are missing, flip these to REQUIRED above.

# --- Your sources (unchanged)
add_executable(Rutgers_ECE451_Parallel
		archive/sessions/01/countPrimesBruteForceSIngleThreaded.cpp
		archive/sessions/01/countupdown.cpp
		archive/sessions/01/countupdownmultithread.cpp
		archive/sessions/01/countupdownmultithreadmutex.cpp
		archive/sessions/01/funcs.cpp
		archive/sessions/01/hintsformultithreadedhw.cpp
		archive/sessions/02/eratosthenes.cpp
		archive/sessions/02/hwmandelbrot.cpp
		archive/sessions/02/scalar_vs_avx2.cpp
		archive/sessions/02/sorting_lookahead.cpp
		archive/sessions/03/avxhomeworkissuse.cpp
		archive/sessions/03/examining_intrinsics.cpp
		archive/sessions/03/examining_intrinsics_main.cpp
		archive/sessions/03/mergesort.cpp
		archive/sessions/03/sortinghomework.cpp
		archive/sessions/04/memory_access.cpp
		archive/sessions/05/hw_matmult.cpp
		archive/sessions/05/openmp.cpp
		archive/sessions/05/openmpsum.cpp
		archive/sessions/06/FiniteElement.cpp
		archive/sessions/06/mandel_unified_clean/main.cpp
		archive/sessions/06/mandel_unified_clean/mandelbrot_serial.cpp
		archive/sessions/06/mandel_unified_clean/mandelbrot_tasks_serial.cpp
		archive/sessions/06/mandel_unified_clean/tasksys.cpp
		archive/sessions/06/mandel_unified_clean/timing.h
		archive/sessions/06/mandelbrot/mandelbrot.cpp
		archive/sessions/06/mandelbrot/mandelbrot_serial.cpp
		archive/sessions/06/mandelbrot_tasks/mandelbrot_tasks.cpp
		archive/sessions/06/mandelbrot_tasks/mandelbrot_tasks_serial.cpp
		archive/sessions/06/mandelbrot_tasks/tasksys.cpp
		archive/sessions/09/sort.cu
		archive/sessions/10/mpi_advection.cpp
		archive/sessions/11/gameoflife.cpp
		archive/sessions/11/gameoflife/gameoflife_mpi.cpp
		archive/sessions/11/gameoflife_bitwise.cpp
		archive/sessions/14/benchmark_threaded_memory.cc
		archive/sessions/14/test.cpp
		exam-review/final/01_omp_parallel.cpp
		exam-review/final/02openmp.cpp
		exam-review/final/03final.cu
		exam-review/test1/avx_instructions_test1.cpp
		exam-review/test1/memorytimings.cpp
		exam-review/test1/simd.cpp
		exam-review/test1/unoptimized_code.cpp
		exam-review/test2/mult.cu
		homework/HW0.cpp
		references/c++/02Matrix.cpp
		references/c++/03Matrix.cpp
		references/c++/04Matrix.cpp
		references/c++/Matrix.cpp
		references/c++/storagespecifiers.cpp
		sessions/01/00first.cpp
		sessions/01/01threads.cpp
		sessions/01/02account.cpp
		sessions/01/03account_synchro.cpp
		sessions/02/00multithread_prime_brute.cpp
		sessions/02/03eratosthenes.cpp
		sessions/02/04_hopefullyworking_primes.cpp
		sessions/02/05_testing_optimization.cpp
		sessions/02/assembly_overview.cpp
		sessions/02/examplesum.cpp
		sessions/03/00_improved_eratosthenes.cpp
		sessions/03/00_simple_eratosthenes.cpp
		sessions/03/01_better_eratosthenes.cpp
		sessions/03/02_bitwise_eratosthenes.cpp
		sessions/03/03_racecondition.cpp
		sessions/03/04_synchronization.cpp
		sessions/03/05_dict.cpp
		sessions/03/hw3_countwords.cpp
		sessions/03/matrix.cpp
		sessions/03/primebits.cpp
		sessions/03/triephrases.cpp
		sessions/05/00simd.cpp
		sessions/05/01_dotproduct.cpp
		sessions/05/euler.cpp
		sessions/05/void_mandel_ex.cpp
		sessions/06/benchmark_assembly_mem.cpp
		sessions/06/benchmarks.cpp
		sessions/07/00b_mandelbrot_hw_hints.cpp
		sessions/07/00mandelbrot.cpp
		sessions/07/01testsimd.cpp
		sessions/07/02sortingnetwork.cpp
		sessions/07/03vectorization.cpp
		sessions/07/ExampleProblems.md.cpp
		sessions/07/scope.cpp
		sessions/07/test_intrinsics.cpp
		sessions/08/memory_perf.cpp
		sessions/09/memfuncs.cpp
		sessions/09/memorybenchmarks.cpp
		sessions/10/04matrix_styles.cpp
		sessions/10/benchmark.hpp
		sessions/10/gravsim0.cpp
		sessions/12/demo_omp.cpp
		sessions/13/SolarSystem2.cpp
		sessions/13/solar_system.cpp
		sessions/14/cplusplus_naming_rules.cpp
		sessions/14/floating_point.cpp
		sessions/14/merge.cpp
		sessions/14/src/01parallel_block.cpp
		sessions/14/src/01parallel_block_threadsafe.cpp
		sessions/14/src/02_sum_simd.cpp
		sessions/14/src/03_dot_simd.cpp
		sessions/14/src/05_displayopenmp.cpp
		sessions/14/src/06_scheduling_control.cpp
		sessions/14/src/benchmark.hpp
		sessions/14/src/simplesort.cu
		sessions/14/src/threadreview.cpp
		sessions/15/03_cuda.cu
		sessions/15/04_eratosthenes_bad.cu
		sessions/15/blur.cu
		sessions/15/cuda_sum.cu
		sessions/15/displaycapabilities.cu
		sessions/15/matmul.cu
		sessions/15/simplesort.cu
		sessions/15/squares.cu
		sessions/17/GameOfLife.cpp
		sessions/17/GameOfLife_mpi.cpp
		sessions/19/GameOfLife_mpi.cpp
)

# You previously had a global include for this folder; make it target-local:
target_include_directories(Rutgers_ECE451_Parallel PRIVATE
		archive/sessions/06/mandel_unified_clean
)

# --- SIMD/AVX flags (your code uses AVX/AVX2 intrinsics)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	target_compile_options(Rutgers_ECE451_Parallel PRIVATE
			$<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
	)
endif ()

# --- CUDA settings (only if Toolkit found)
if (CUDAToolkit_FOUND)
	# Use a sensible default arch; adjust if you like
	# Example for Ampere+; you can also set CMAKE_CUDA_ARCHITECTURES on command line.
	if (NOT CMAKE_CUDA_ARCHITECTURES)
		set(CMAKE_CUDA_ARCHITECTURES 75;86;89 CACHE STRING "" FORCE)
	endif ()

	# CUDA runtime
	target_link_libraries(Rutgers_ECE451_Parallel PRIVATE CUDA::cudart)
	# Nice defaults for .cu compilation
	set_target_properties(Rutgers_ECE451_Parallel PROPERTIES
			CUDA_STANDARD 17
			CUDA_SEPARABLE_COMPILATION ON
	)
endif ()

# --- OpenMP (if available)
if (OpenMP_CXX_FOUND)
	target_link_libraries(Rutgers_ECE451_Parallel PRIVATE OpenMP::OpenMP_CXX)
endif ()

# --- MPI (if available)
if (MPI_CXX_FOUND)
	target_link_libraries(Rutgers_ECE451_Parallel PRIVATE MPI::MPI_CXX)
endif ()

# --- WebP (if available; for <webp/encode.h>)
if (WEBP_FOUND)
	target_link_libraries(Rutgers_ECE451_Parallel PRIVATE PkgConfig::WEBP)
else ()
	message(WARNING "libwebp not found via pkg-config; WebP features will be disabled")
endif ()

# --- Threads for std::thread / pthread
target_link_libraries(Rutgers_ECE451_Parallel PRIVATE Threads::Threads)

# --- Faster builds for release
if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()
