ompcode:
	g++ -O3 -g -fopenmp -mavx2 -mfma ompcode.cpp -oompcode
	g++ -O3 -S -fopenmp -mavx2 -mfma ompcode.cpp
# the compiler will use AVX2 registers anyway, no guarantee that #pragma omp simd will use AVX2
# will be any better than the compiler itself. We can try without -mavx2 to see
# what those vector registers do for us...

fractalfern: fractalfern.cpp
	g++ -O3 -g -fopenmp -mavx2 -mfma fractalfern.cpp -o fractalfern -lwebp

fractalfern_omp: fractalfern_omp.cpp
	g++ -O3 -g -fopenmp -mavx2 -mfma fractalfern_omp.cpp -o fractalfern_omp -lwebp

