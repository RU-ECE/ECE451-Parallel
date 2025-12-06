#include <malloc.h>
#include <memory.h>

using namespace std;

void printvadd_andload(const unsigned int* ap, const unsigned int* bp);

int main() {
	const auto a = static_cast<unsigned int*>(memalign(32, 32)), b = static_cast<unsigned int*>(memalign(32, 32));
	const unsigned int aorig[] = {1, 2, 3, 4, 5, 6, 7, 8}, borig[] = {3, 5, 9, 2, 1, 4, 2, 1};
	memcpy(a, aorig, 32);
	memcpy(b, borig, 32);

	printvadd_andload(a, b);
}
