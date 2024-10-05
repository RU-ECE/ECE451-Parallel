<a href="https://www.ece.rutgers.edu">
<img src="assets/RUshield.png" alt="Rutgers Logo" width="100">
</a>

# ECE-451/556 Parallel and Distributed Computing

* 2024 Fall
* **Instructor:**  [Dov Kruger](https://www.ece.rutgers.edu/Dov-Kruger)
* **[Meeting Times and Office hours](https://bit.ly/3ObwKEr)**
* **[Resources](ref)**
* **[Course Web Address](https://github.com/RU-ECE/ECE451-Parallel)**

**Prerequisites**

  * Required good command of C++, helpful to know some machine language but not required
  * 16:332:563 (Grads) or 16:332:331 (undergrads)
  * 16:332:351 (Programming Methodology-II & Data Structures)
  * Strongly advised to have a solid background using Linux
    * Tools in this course will use linux, remote machines will be available
  * 14:332:434 – Operating Systems (helpful)

## Software and Hardware Requirements

Because students today may have a Mac running an ARM CPU, we will provide online facilities, but if you can compile on your own intel machine this will lighten the load on the central resources. Everyone is expected to get an account on Amarel, the Rutgers research cluster, and to use the resource carefully as we are guests.

* g++/clang++ compiler capable of executing c++-14 and OpenMP
* c++ threading
* Intel 4th generation or better capable of executing AVX2 instructions
* CUDA running on NVIDIA GTX 10xx or better
* OpenMPI (running on Amarel cluster)
* Icarus Verilog (if time permits)

## COURSE DESCRIPTION

This course covers parallel computing

* Overview of computer architecture and speed
  * Current limits to parallel execution speed
    * Memory bandwidth
    * Latency
    * Scheduling
    * Load balancing
  * Future Improvements
* Approaches to Parallelism
  * Multithreading
  * vectorization
  * Optimization of memory bandwidth
  * Custom hardware architecture
* Technologies
  * C++ threads
    * locking
  * Vectorization (AVX2, AVX512 or Neon)
  * OpenMP (multithreaded and vectorized)
  * CUDA (massively parallel execution on GPUs)
  * MPI (Message Passing Interface)
  * Verilog (custom parallel computing if time permits)

## Course Outcomes

After completion of this course, students will be able to
*  Write code using C++ threads
*  Write code using openmp
*  Benchmark code to measure actual performance
*  Parallelize algorithms
*  Analyze problems to determine whether they are memory bound
*  Optimize algorithms to decrease memory utilization where possible 
*  Write SIMD programs using AVX instructions or ARM Neon
*  Use a parallelizing compiler like OpenMP to generate vectorized code
*  Write CUDA kernels to run on NVIDIA GPUs
*  Write parallel programs on clusters using MPI
*  Identify performance bottlenecks in current computing architectures
## FORMAT AND STRUCTURE

* Classes include slides and live coding in a number of C/C++ derivative languages and APIs. You are encouraged to actively participate.

## COURSE MATERIALS

There is no required textbook. Most materials listed here are freely available.
There are some books that are helpful
* [multithreading tutorialpoint]https://www.tutorialspoint.com/cplusplus/cpp_multithreading.htm
* [multithreading geeksforgeeks](https://www.geeksforgeeks.org/multithreading-in-cpp/)
* [c++ atomic](https://www.freecodecamp.org/news/atomics-and-concurrency-in-cpp/)
* [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
* [Felix Cloutier Intel Assembler Docs](https://www.felixcloutier.com/x86/)
* Optional textbooks
  * Programming Massively Parallel Processors 4e, Hwu, Kirk, and Hajj, 2023.
  * 
* [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [OpenMP Manual](ref/OpenMP-API-Specification-5.0.pdf)
* [Intel ISPC Compiler](https://ispc.github.io/)

* Other Readings: 	Papers available in ref directory of repo

## COURSE REQUIREMENTS
* **Attendance:**	Attendance is crucial for an effective learning but will not be graded. There will be in-class handouts to encourage participation and attendance. Not doing the in-class work means you are not eligible for curving or grade modifications.
* **Homework:** 	Coding assignments will be submitted via canvas for individual single files, or via github. Theory assignments will be submitted on paper or in canvas quizzes.

## GRADING PROCEDURES

Grades will be based on:
* Homework problem sets on theory                     (5%)
* Paired Programming Homeworks                        (15%)
* Mini projects                                       (15%)
* Test 1                                              (16.25%)
* Test 2                                              (16.25%)
* Final exam                                          (32.5%)

[Grading Policies] (https://github.com/RU-ECE/DovKrugerCourses/grading.md)
[Academic Honesty and Discipline] (https://github.com/RU-ECE/DovKrugerCourses/academichonesty.md)

## IMPORTANT DATES
* Test 1           ** 2024-TBD **
* Test 2           ** 2024-TBD **
* Final            ** 2024-TBD **
