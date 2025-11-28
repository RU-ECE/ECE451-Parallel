#include <cstdint>
#include <malloc.h>
#include <memory.h>

using namespace std;

void printvadd_andload(const uint32_t* ap, const uint32_t* bp);

int main() {
	const auto a = static_cast<uint32_t*>(memalign(32, 32));
	const auto b = static_cast<uint32_t*>(memalign(32, 32));
	const uint32_t aorig[] = {1, 2, 3, 4, 5, 6, 7, 8};
	const uint32_t borig[] = {3, 5, 9, 2, 1, 4, 2, 1};
	memcpy(a, aorig, 32);
	memcpy(b, borig, 32);

	printvadd_andload(a, b);
}
