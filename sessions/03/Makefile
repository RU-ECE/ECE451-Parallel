examining_intrinsics: examining_intrinsics.cpp examining_intrinsics_main.cpp
	g++ -O2 -mavx2 examining_intrinsics.cpp examining_intrinsics_main.cpp -o examining_intrinsics

hwsort8col: sortinghomework.cpp sortinghomework_asm.s
	g++ -O2 -mavx2 sortinghomework.cpp sortinghomework_asm.s -o $@

.PHONY: clean

clean:
	rm -f examining_intrinsics