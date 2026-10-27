raytrace: main.cc camera.h hittable.h hittable_list.h material.h sphere.h rtweekend.h
	g++ -std=c++11 -g -O3 -o raytrace $< -lwebp
	g++ -std=c++11 -mavx2 -fopenmp -g -O3 -o raytrace_omp $< -lwebp

clean:
	rm raytrace

