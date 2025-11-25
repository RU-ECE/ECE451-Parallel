cmake_minimum_required(VERSION 3.22)

# Enable CUDA as a first-class language (for .cu files)
project(Rutgers_ECE451_Parallel LANGUAGES CXX CUDA)

# C++23 for all C++ sources
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
		references/c++/02Matrix.cpp
		references/c++/03Matrix.cpp
		references/c++/04Matrix.cpp
		references/c++/Matrix.cpp
		references/c++/storagespecifiers.cpp
		sessions/01/00first.cpp
		sessions/01/01threads.cpp
		sessions/01/02account.cpp
		sessions/01/03account_synchro.cpp
		homework/HW0.cpp
		sessions/02/00multithread_prime_brute.cpp
		sessions/02/03eratosthenes.cpp
		sessions/02/04_hopefullyworking_primes.cpp
		sessions/02/05_testing_optimization.cpp
		sessions/02/assembly_overview.cpp
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
		"sessions/05/void mandel_ex.cpp"
		sessions/06/benchmark_assembler_mem.cpp
		sessions/06/benchmarks.cpp
		sessions/07/00b_mandelbrot_hw_hints.cpp
		sessions/07/00mandelbrot.cpp
		sessions/07/01testsimd.cpp
		sessions/07/02sortingnetwork.cpp
		sessions/07/03vectorization.cpp
		sessions/07/ExampleProblems.md.cpp
		sessions/07/test_intrinsics.cpp
		sessions/08/memory_perf.cpp
		sessions/09/memfuncs.cpp
		sessions/09/memorybenchmarks.cpp
		sessions/10/04matrix_styles.cpp
		sessions/10/benchmark.hpp
		sessions/10/gravsim0.cpp
		sessions/12/demo_omp.cpp
		sessions/13/solar_system.cpp
		sessions/13/SolarSystem2.cpp
		sessions/14/src/01parallel_block.cpp
		sessions/14/src/01parallel_block_threadsafe.cpp
		sessions/14/src/02_sum_simd.cpp
		sessions/14/src/03_dot_simd.cpp
		sessions/14/src/05_displayopenmp.cpp
		sessions/14/src/06_scheduling_control.cpp
		sessions/14/src/benchmark.hpp
		sessions/14/src/threadreview.cpp
		sessions/14/src/simplesort.cu
		sessions/14/cplusplus_naming_rules.cpp
		sessions/14/floating_point.cpp
		sessions/14/merge.cpp
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
		sessions/old/01/countPrimesBruteForceSIngleThreaded.cpp
		sessions/old/01/countupdown.cpp
		sessions/old/01/countupdownmultithread.cpp
		sessions/old/01/countupdownmultithreadmutex.cpp
		sessions/old/01/funcs.cpp
		sessions/old/01/hintsformultithreadedhw.cpp
		sessions/old/02/eratosthenes.cpp
		sessions/old/02/hwmandelbrot.cpp
		sessions/old/02/scalar_vs_avx2.cpp
		sessions/old/02/sorting_lookahead.cpp
		sessions/old/03/avxhomeworkissuse.cpp
		sessions/old/03/examining_intrinsics.cpp
		sessions/old/03/examining_intrinsics_main.cpp
		sessions/old/03/mergiesort..cpp
		sessions/old/03/sortinghomework.cpp
		sessions/old/04/memory_access.cpp
		sessions/old/05/hw_matmult.cpp
		sessions/old/05/openmp.cpp
		sessions/old/05/openmpsum.cpp
		sessions/old/06/mandel_unified_clean/main.cpp
		sessions/old/06/mandel_unified_clean/mandelbrot_serial.cpp
		sessions/old/06/mandel_unified_clean/mandelbrot_tasks_serial.cpp
		sessions/old/06/mandel_unified_clean/tasksys.cpp
		sessions/old/06/mandel_unified_clean/timing.h
		sessions/old/06/mandelbrot/mandelbrot.cpp
		sessions/old/06/mandelbrot/mandelbrot_serial.cpp
		sessions/old/06/mandelbrot_tasks/mandelbrot_tasks.cpp
		sessions/old/06/mandelbrot_tasks/mandelbrot_tasks_serial.cpp
		sessions/old/06/mandelbrot_tasks/tasksys.cpp
		sessions/old/06/FiniteElement.cpp
		sessions/old/09/sort.cu
		sessions/old/10/mpi_advection.cpp
		sessions/old/11/gameoflife/gameoflife_mpi.cpp
		sessions/old/11/gameoflife.cpp
		sessions/old/11/gameoflife_bitwise.cpp
		sessions/old/14/benchmark_threaded_memory.cc
		sessions/old/14/test.cpp
		sessions/reviewfinal/01_omp_parallel.cpp
		sessions/reviewfinal/02openmp.cpp
		sessions/reviewfinal/03final.cu
		sessions/reviewtest1/avx_instructions_test1.cpp
		sessions/reviewtest1/memorytimings.cpp
		sessions/reviewtest1/simd.cpp
		sessions/reviewtest1/unoptimized_code.cpp
		sessions/reviewtest2/mult.cu
)

# You previously had a global include for this folder; make it target-local:
target_include_directories(Rutgers_ECE451_Parallel PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/sessions/old/06/mandel_unified_clean
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
