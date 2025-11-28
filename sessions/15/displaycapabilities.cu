/*****************************************************************************
 * Display the capabilities of the GPU this program runs on
 ****************************************************************************/
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <time.h>

#include "cuda_runtime.h"

using namespace std;
using u32 = uint32_t;
using u64 = uint64_t;
using prof_info = struct {
	float seconds;
};

timespec timer_start() {
	timespec start_time;
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	return start_time;
}

// Call this function on the return value of timer_start to get the total
// number of nanoseconds that elapsed.
long timer_end(const timespec start_time) {
	timespec end_time;
	clock_gettime(CLOCK_MONOTONIC, &end_time);
	long diffInNanos =
		1'000'000'000 * end_time.tv_sec + end_time.tv_nsec - (1'000'000'000 * start_time.tv_sec + start_time.tv_nsec);
	if (end_time.tv_nsec - start_time.tv_nsec < 0) {
		diffInNanos = 1'000'000'000 * (end_time.tv_sec + 1) + end_time.tv_nsec -
			(1'000'000'000 * (start_time.tv_sec - 1) + start_time.tv_nsec);
	}
	return diffInNanos;
}


int global;

void f() {
	for (auto i = 0; i < 1'000'000; i++)
		global = (global << 1) + 17; // just compute something so the compiler has to do this.
}
const char* powersk[] = {"k", "M", "G", "T"};
string powerk(u64 v) {
	u32 power = 0;
	while (v > 0 && v % 1024 == 0) {
		power++;
		v /= 1024;
	}
	return to_string(v) + powersk[power];
}
#define check_status(st)                                                                                               \
	if (st != 0) {                                                                                                     \
		cerr << "Error file: " << __FILE__ << " line: " << __LINE__;                                                   \
		return;                                                                                                        \
	}
#define dispIntProp(sym) left << setw(27) << #sym << powerk(prop.sym) << endl <<
#define dispProp(sym) left << setw(27) << #sym << prop.sym << endl <<
#define dispProp3D(sym) left << setw(27) << #sym << prop.sym[0] << ", " << prop.sym[1] << ", " << prop.sym[2] << endl <<

/*
 * name = "NVIDIA GeForce GTX 1070",
 * uuid = {bytes = "\311Z\243\261\370Sw|\247\257k\346\vZ\361N",
 * luid = "\000\000\000\000\000\000\000",
 * luidDeviceNodeMask = 0,
 * totalGlobalMem = 8513978368,
 * sharedMemPerBlock = 49152,
 * regsPerBlock = 65536,
 * warpSize = 32,
 * memPitch = 2147483647,
 * maxThreadsPerBlock = 1024,
 * maxThreadsDim = {1024, 1024, 64},
 * maxGridSize = {2147483647, 65535, 65535},
 * clockRate = 1784500,
 * totalConstMem = 65536,
 * major = 6, minor = 1,
 * textureAlignment = 512,
 * texturePitchAlignment = 32,
 * deviceOverlap = 1,
 * multiProcessorCount = 15,
 * kernelExecTimeoutEnabled = 0,
 * integrated = 0,
 * canMapHostMemory = 1,
 * computeMode = 0,
 * maxTexture1D = 131072,
 * maxTexture1DMipmap = 16384,
 * maxTexture1DLinear = 268435456,
 * maxTexture2D = {131072, 65536},
 * maxTexture2DMipmap = {32768, 32768},
 * maxTexture2DLinear = {131072, 65000, 2097120},
 * maxTexture2DGather = {32768, 32768},
 * maxTexture3D = {16384, 16384, 16384},
 * maxTexture3DAlt = {8192, 8192, 32768},
 * maxTextureCubemap = 32768,
 * maxTexture1DLayered = {32768, 2048},
 * maxTexture2DLayered = {32768, 32768, 2048},
 * maxTextureCubemapLayered = {32768, 2046},
 * maxSurface1D = 32768,
 * maxSurface2D = {131072, 65536},
 * maxSurface3D = {16384, 16384, 16384},
 * maxSurface1DLayered = {32768, 2048},
 * maxSurface2DLayered = {32768, 32768, 2048},
 * maxSurfaceCubemap = 32768,
 * maxSurfaceCubemapLayered = {32768, 2046},
 * surfaceAlignment = 512,
 * concurrentKernels = 1,
 * ECCEnabled = 0,
 * pciBusID = 33,
 * pciDeviceID = 0,
 * pciDomainID = 0,
 * tccDriver = 0,
 * asyncEngineCount = 2,
 * unifiedAddressing = 1,
 * memoryClockRate = 4004000,
 * memoryBusWidth = 256,
 * l2CacheSize = 2097152,
 * persistingL2CacheMaxSize = 0,
 * maxThreadsPerMultiProcessor = 2048,
 * streamPrioritiesSupported = 1,
 * globalL1CacheSupported = 1,
 * localL1CacheSupported = 1,
 * sharedMemPerMultiprocessor = 98304,
 * regsPerMultiprocessor = 65536,
 * managedMemory = 1,
 * isMultiGpuBoard = 0,
 * multiGpuBoardGroupID = 0,
 * hostNativeAtomicSupported = 0,
 * singleToDoublePrecisionPerfRatio = 32,
 * pageableMemoryAccess = 0,
 * concurrentManagedAccess = 1,
 * computePreemptionSupported = 1,
 * canUseHostPointerForRegisteredMem = 1,
 * cooperativeLaunch = 1,
 * cooperativeMultiDeviceLaunch = 1,
 * sharedMemPerBlockOptin = 49152,
 * pageableMemoryAccessUsesHostPageTables = 0,
 * directManagedMemAccessFromHost = 0,
 * maxBlocksPerMultiProcessor = 32,
 * accessPolicyMaxWindowSize = 0,
 * reservedSharedMemPerBlock = 0}
 */
void dump(cudaDeviceProp prop) {
	cout << "Display CUDA device properties\n";
	u64 uuid = *reinterpret_cast<u64*>(&prop.uuid);
	// TODO: UUID is probably being displayed backwards (little-endian?)
	cout << "UUID:               " << hex << uuid << dec << endl
		 << "Name:                      " << prop.name << endl
		 << "Capabilities:              " << prop.major << "." << prop.minor << endl
		 <<
		// TODO: automatically recognize powers of 2 and display 2G, 48M instead of huge numbers?
		dispIntProp(totalGlobalMem) dispIntProp(sharedMemPerBlock) dispIntProp(regsPerBlock) dispIntProp(memPitch)
			dispProp(multiProcessorCount) dispIntProp(sharedMemPerBlock) dispIntProp(totalConstMem) // = 65536,
		dispIntProp(clockRate) // = 1784500,
		dispProp(memoryClockRate) dispProp(memoryBusWidth) dispProp(maxThreadsPerBlock)
			dispIntProp(sharedMemPerMultiprocessor) dispIntProp(regsPerMultiprocessor)
				dispIntProp(persistingL2CacheMaxSize) dispProp(l2CacheSize) dispProp(accessPolicyMaxWindowSize)
					dispProp3D(maxThreadsDim) // = {1024, 1024, 64},
		dispProp3D(maxGridSize) // = {2147483647, 65535, 65535},
		"\n\n\n";
}

int main() {
	cudaError_t st;
	// Send as many threads as possible per block.
	auto cuda_device_ix = 0;
	cudaDeviceProp prop;
	st = cudaGetDeviceProperties(&prop, cuda_device_ix);
	check_status(st);
	int cudaDeviceCount;
	cudaGetDeviceCount(&cudaDeviceCount);
	constexpr auto profile_times = 10;

	dump(prop);

	for (auto i = 0; i < profile_times; i++) {
		prof_info times[profile_times];
		timespec t = timer_start();
		f();
		times[i].seconds = timer_end(t) * 1.0E-9;
	}
	return 0;
}
