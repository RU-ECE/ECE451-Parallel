CXX	:= g++ -O2 -g
mand:	main.cpp  mandelbrot_serial.cpp mandelbrot.ispc mandelbrot_tasks.ispc
	$(CXX) -c mandelbrot_serial.cpp
	$(CXX) -c tasksys.cpp
	ispc --target=avx512 mandelbrot.ispc -o mandelbrot.o -h mandelbrot_ispc.h
#	ispc --target=avx2 mandelbrot.ispc -o mandelbrot.o -h mandelbrot_ispc.h
#	ispc --target=avx2 mandelbrot_tasks.ispc -o mandelbrot_tasks.o -h mandelbrot_tasks_ispc.h
	ispc --target=avx512 mandelbrot.ispc -o mandelbrot.o -h mandelbrot_ispc.h

	$(CXX) main.cpp mandelbrot_serial.o mandelbrot.o mandelbrot_tasks.o tasksys.o -o mand
#-L ~/ispc1.2.0/lib64 -lispcrt_device_cpu
