<a href="https://www.ece.rutgers.edu">
<img src="assets/RUshield.png" alt="Rutgers Logo" width="100">
</a>

# ECE-451/556 Parallel and Distributed Computing
* 2023 Fall
* **Instructor:**  [Dov Kruger](https://RU-ECE/DovKrugerCourses/DovKrugerBio.md)
* **[Meeting Times and Office hours](https://bit.ly/3ObwKEr)
* **[Resources](https://github.com/RU-ece/DovKrugerCourses/DovKrugerBio.md)**
* **[Course Web Address](https://github.com/RU-ECE/ECE451-Parallel)**
** Prerequisites**
  * Required good command of C++, good idea to know some machine language
  * 16:332:563 (Grads) or 16:332:331 (undergrads)
  * 16:332:351 (Programming Methodology-II & Data
  * Strongly advised to have a solid background using Linux
    * Tools in this course will use linux, remote machines will be available
  * 14:332:434 – Operating Systems (would be a plus)

## Hardware Requirements

Because students today may have a Mac running an ARM CPU, we will provide online facilities, but if you can compile on your own intel machine this will lighten the load on the central resources.

* Intel 4th generation or better capable of executing AVX2 instructions
* Intel ISPC compiler
* CUDA running on NVIDIA GTX 10xx or better

## COURSE DESCRIPTION

This course covers parallel computing

* Overview of computer architecture and speed
* Approaches to Parallelism
* Current limits to parallel execution speed
  * Memory bandwidth
  * Latency

## Course Outcomes

After completion of this course, students will be able to
*  Write code using C++ threads
*  Write code using openmp
*  Benchmark code to measure actual performance
*  Parallelize algorithms
*  Analyze problems to determine whether they are memory bound
*  Optimize algorithms to decrease memory utilization where possible 
*  Write SIMD programs using AVX instructions (or ARM Neon if you prefer)
*  Use a parallelizing compiler like ISPC to generate vectorized code 
*  Write CUDA kernels to run on NVIDIA GPUs
*  Write parallel programs on clusters using MPI

## FORMAT AND STRUCTURE
* Classes include slides and live coding. You are encouraged to actively participate.

## COURSE MATERIALS

All textbooks are optional. Most materials for this course are linked in the notes and are freely downloadable.

* Optional textbooks
  * Parallel Computer Architecture: a Hardware/Software approach, David E. Culler, Jaswinder Pal Singh, Anoop Gupta. Morgan Kaufman, ISBN 558603433
* [Intel ISPC Compiler](https://ispc.github.io/)
* [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
* [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* 

* Other Readings: 	Papers available in ref directory of repo

## COURSE REQUIREMENTS
* **Attendance:**	Attendance is crucial for an effective learning but will not be graded. There will be in-class handouts to encourage participation and attendance. Not doing the in-class work means you are not eligible for curving or grade modifications.
* **Homework:** 	Coding assignments will be submitted via canvas for individual single files, or via github. Theory assignments will be submitted on paper or in canvas quizzes.

## GRADING PROCEDURES
Grades will be based on:
* Homework problem sets on theory                      (5%)
* Group Programming Homeworks                         (20%)
* Mini Project                                        (15%)
* Midterm                                             (30%)
* Final Project or final exam                         (30%)

[Grading Policies] (https://github.com/RU-ECE/DovKrugerCourses/grading.md)
[Academic Honesty and Discipline] (https://github.com/RU-ECE/DovKrugerCourses/academichonesty.md)

## IMPORTANT DATES
* Midterm          ** 2023-TBD **
* Final Project    ** 2023-TBD **
